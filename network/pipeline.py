import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd

from .MaskFlownet import *
from .config import Reader
from .layer import Reconstruction2D, Reconstruction2DSmooth

import cv2

def build_network(name):
    return eval(name)

class PipelineFlownet:
    _lr = None

    def __init__(self, weighting_factor, ctx, config):
        self.ctx = ctx
 
        self.network = build_network(getattr(config.network, 'class').get('MaskFlownet'))(weighting_factor=weighting_factor,
                                                                                          config=config)
        self.network.hybridize()
        self.network.collect_params().initialize(init=mx.initializer.MSRAPrelu(slope=0.1), ctx=self.ctx)
        self.trainer = gluon.Trainer(self.network.collect_params(), 'adam', {'learning_rate': 1e-4, 'clip_gradient': 10})
        self.strides = self.network.strides or [64, 32, 16, 8, 4]

        self.scale = self.strides[-1]
        self.upsampler = Upsample(self.scale)
        self.upsampler_mask = Upsample(self.scale)

        self.epeloss = EpeLoss()
        self.epeloss.hybridize()
        self.epeloss_with_mask = EpeLossWithMask()
        self.epeloss_with_mask.hybridize()

        multiscale_weights = config.network.mw.get([.005, .01, .02, .08, .32])
        if len(multiscale_weights) != 5:
            multiscale_weights = [.005, .01, .02, .08, .32]
            
        self.multiscale_epe = MultiscaleEpe(
            scales = self.strides, weights = multiscale_weights, match = 'upsampling',
            eps = 1e-8, q = config.optimizer.q.get(None))
        self.multiscale_epe.hybridize()

        self.multiscale_disp_epe = MultiscaleEpeDisp(
            scales = self.strides, weights = multiscale_weights, match = 'upsampling',
            eps = 1e-8, q = config.optimizer.q.get(None))
        self.multiscale_epe.hybridize()

        self.lr_schedule = config.optimizer.learning_rate.value
        
        print('GPU Devices:           ', ctx) 
        print('Initialize network:    ', getattr(config.network, 'class').get('MaskFlownet'))
        print('Use weighting factor:  ', weighting_factor)
    
    def intialize_coords_pyramid(self, batch_size, target_shape):
        print('Initialize coords pyramid...')
        self.coords_pyramid = self.create_coords_pyramid(batch_size, target_shape) 
        
    def save(self, prefix):
        self.network.save_parameters(prefix + '.params')
        self.trainer.save_states(prefix + '.states')

    def load(self, checkpoint):
        self.network.load_parameters(checkpoint, ctx=self.ctx)

    def load_head(self, checkpoint):
        self.network.load_head(checkpoint, ctx=self.ctx)

    def fix_head(self):
        self.network.fix_head()
    
    def set_learning_rate(self, steps):
        i = 0
        while i < len(self.lr_schedule) and steps > self.lr_schedule[i][0]:
            i += 1
        try:
            lr = self.lr_schedule[i][1]
        except IndexError:
            return False

        self.trainer.set_learning_rate(lr)
        self._lr = lr
        
        return True 

    @property
    def lr(self):
        return self._lr

    def loss(self, pred, occ_masks, labels, masks):
        loss = self.multiscale_epe(labels, masks, *pred)
        return loss

    def displacement_aware_loss(self, cpred, occ_masks, labels, masks, disp_masks):
        disp_loss = self.multiscale_disp_epe(labels, masks, disp_masks, *cpred)

        return disp_loss

    def centralize(self, img1, img2):
        rgb_mean = nd.concat(img1, img2, dim = 2).mean(axis = (2, 3)).reshape((-2, 1, 1))
        return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

    def generate_weighted_disp_masks(self, flow, _LARGE_DISP):
        flow_mag = nd.norm(flow, axis=1, keepdims=True)
        flow_mag = nd.broadcast_div(flow_mag, _LARGE_DISP)

        flow_mag = nd.broadcast_minimum(flow_mag, nd.ones((1), ctx=flow_mag.context))
        
        small_disp_masks = 1.0 - flow_mag
        large_disp_masks = flow_mag

        stacked_mask = nd.concat(small_disp_masks, large_disp_masks, dim=1)
        
        return stacked_mask

    def create_coords_pyramid(self, batch_size, shape, n_blocks=4, n_lvls=[4, 4, 3, 2], r=4):
        ori_h, ori_w = shape
        
        h = ori_h // 4
        w = ori_w // 4

        dx = nd.arange(-r, r+1).reshape((1, 2*r+1)).repeat(2*r+1, axis=0)
        dy = nd.arange(-r, r+1).reshape((1, 2*r+1)).repeat(2*r+1, axis=1)
        
        delta = nd.stack(dx, dy, axis=-1)
        delta_lvl = delta.reshape((1, 2*r+1, 2*r+1, 2))

        coords_pyramid = []
        for i in range(n_blocks): # from h/4 (block2 ~ block6) h/64
            # x-y indexing
            gx = nd.arange(w).reshape((1, w)).repeat(h, axis=0)
            gy = nd.arange(h).reshape((h, 1)).repeat(w, axis=1)

            coords = nd.stack(gx, gy, axis=-1)
            coords = nd.expand_dims(coords, axis=0)
            coords = nd.tile(coords, (batch_size, 1, 1, 1))
            coords = nd.reshape(coords, (-1, 1, 1, 2))

            n_lvl = n_lvls[i]
            for j in range(n_lvl):
                centroid_lvl  = coords / 2**j
                coords_lvl = nd.broadcast_add(centroid_lvl, delta_lvl)
                coord_x, coord_y = coords_lvl.split(axis=-1, num_outputs=2)
                
                # normalized, w need to be divided
                coord_x = 2*coord_x/((w / 2**j)-1) -1
                coord_y = 2*coord_y/((h / 2**j)-1) -1
                coord = nd.concat(coord_x, coord_y, dim=-1)
                coord = nd.transpose(coord, (0, 3, 1, 2))

                coords_pyramid.append(coord)

            h, w = h//2, w//2

        return coords_pyramid

    def train_batch(self, img1, img2, label, geo_aug, color_aug, mask = None):
        losses = []
        disp_losses = []
        combined_losses = []
        epes = []

        batch_size = img1.shape[0]
        if mask is None:
            mask = np.full(shape = (batch_size, 1, 1, 1), fill_value = 255, dtype = np.uint8)
        
        # Calcuate TAU for separating two branches
        h, w = img1.shape[2:]
        _TAU = np.sqrt((h/5)**2 + (w/5)**2)
        _TAU = np.full(shape=(batch_size, 1, 1, 1), fill_value=_TAU, dtype=np.float32)

        img1, img2, label, mask, _TAU = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, label, mask, _TAU))

        with autograd.record():
            for img1s, img2s, labels, masks, _TAUs in zip(img1, img2, label, mask, _TAU):
                img1s, img2s, labels, masks = img1s / 255.0, img2s / 255.0, labels.astype("float32", copy=False), masks / 255.0
                img1s, img2s, labels, masks = geo_aug(img1s, img2s, labels, masks)

                img1s, img2s = color_aug(img1s, img2s)
                img1s, img2s, _ = self.centralize(img1s, img2s)

                labels = labels.flip(axis = 1)
                disp_masks = self.generate_weighted_disp_masks(labels, _TAUs)
                
                # Move coords to GPU
                coords_pyramid = [nd.array(x, ctx=img1s.context) for x in self.coords_pyramid]

                pred, cpred, occ_masks, disp_mask_preds = self.network(img1s, img2s, _TAUs / 20.0, *coords_pyramid)

                loss = self.loss(pred, occ_masks, labels, masks)
                disp_loss = self.displacement_aware_loss(cpred, occ_masks, labels, masks, disp_masks)
              
                combined_loss = loss + 0.4 * disp_loss
                
                epe = self.epeloss_with_mask(self.upsampler(pred[-1]), labels, masks)

                losses.append(loss)
                disp_losses.append(disp_loss)
                combined_losses.append(combined_loss)
                epes.append(epe)

        avg_epe = np.mean(np.concatenate([epe.asnumpy() for epe in epes]))
        avg_loss = np.mean(np.concatenate([loss.asnumpy() for loss in losses]))
        avg_disp_loss = np.mean(np.concatenate([loss.asnumpy() for loss in disp_losses]))
        avg_combined_loss = np.mean(np.concatenate([loss.asnumpy() for loss in combined_losses]))

        if np.isnan(avg_epe):
            print('NAN! Skip this iteration.')
            
            return {"epe": -1, 'loss': -1}

        for loss in combined_losses:
            loss.backward()

        self.trainer.step(batch_size)

        return {"epe": avg_epe, 'loss': avg_loss, 'disp_loss': avg_disp_loss, 'combined_loss': avg_combined_loss}

    def do_batch_mx(self, img1, img2, resize = None, _TAU=None):
        ''' do a batch of samples range [0,1] with network preprocessing and padding
        '''
        img1, img2, _ = self.centralize(img1, img2)
        shape = img1.shape
        if resize is None:
            pad_h = (64 - shape[2] % 64) % 64
            pad_w = (64 - shape[3] % 64) % 64
        else:
            pad_h = resize[0] - shape[2]
            pad_w = resize[1] - shape[3]
        if pad_h != 0 or pad_w != 0:
            img1 = nd.contrib.BilinearResize2D(img1, height = shape[2] + pad_h, width = shape[3] + pad_w)
            img2 = nd.contrib.BilinearResize2D(img2, height = shape[2] + pad_h, width = shape[3] + pad_w)
        
        coords_pyramid = self.create_coords_pyramid(img1.shape[0], img1.shape[2:]) 
        coords_pyramid = [nd.array(x, ctx=img1.context) for x in coords_pyramid]

        pred, _, _, disp_mask_preds = self.network(img1, img2, _TAU / 20.0, *coords_pyramid)
        
        return pred, disp_mask_preds
       
    def do_batch(self, img1, img2, label = None, mask = None, resize = None, _TAU=None):
        flows, disp_mask_preds = self.do_batch_mx(img1, img2, resize=resize, _TAU=_TAU)
        
        flow = self.upsampler(flows[-1])
        
        shape = img1.shape
        if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
            flow = nd.contrib.BilinearResize2D(flow, height = shape[2], width = shape[3]) * nd.array(
                [shape[d] / flow.shape[d] for d in (2, 3)], ctx = flow.context).reshape((1, 2, 1, 1))
 
        epe = None
        if label is not None and mask is not None:
            epe = self.epeloss_with_mask(flow, label, mask)
        
        return flow, epe, disp_mask_preds

    def validate(self, img1, img2, label, mask = None, batch_size = 1, resize = None, return_type = 'epe'):
        ''' validate the whole dataset
        '''
        np_epes = []
        size = len(img1)
        bs = batch_size
        if mask is None:
            mask = [np.full(shape = (384, 512, 1), fill_value = 255, dtype = np.uint8)] * size

        for j in range(0, size, bs):
            batch_img1 = img1[j: j + bs]
            batch_img2 = img2[j: j + bs]
            batch_label = label[j: j + bs]
            batch_mask = mask[j: j + bs]

            batch_img1 = np.transpose(np.stack(batch_img1, axis=0), (0, 3, 1, 2))
            batch_img2 = np.transpose(np.stack(batch_img2, axis=0), (0, 3, 1, 2))
            batch_label = np.transpose(np.stack(batch_label, axis=0), (0, 3, 1, 2))
            batch_mask = np.transpose(np.stack(batch_mask, axis=0), (0, 3, 1, 2))

            # Generate displacement maskss
            b, _, h, w =  batch_img1.shape
            _TAU = np.sqrt((h/5)**2 + (w/5)**2)
            _TAU = np.full(shape=(b, 1, 1, 1), fill_value=_TAU, dtype=np.float32)

            def Norm(x):
                return nd.sqrt(nd.sum(nd.square(x), axis = 1, keepdims = True))

            batch_epe = []
            ctx = self.ctx[ : min(len(batch_img1), len(self.ctx))]
            nd_img1, nd_img2, nd_label, nd_mask, _TAU = map(lambda x : gluon.utils.split_and_load(x, ctx, even_split = False), (batch_img1, batch_img2, batch_label, batch_mask, _TAU))
            for img1s, img2s, labels, masks, _TAUs in zip(nd_img1, nd_img2, nd_label, nd_mask, _TAU):
                img1s, img2s, labels, masks = img1s / 255.0, img2s / 255.0, labels.astype("float32", copy=False), masks / 255.0
                labels = labels.flip(axis = 1)

                flows, epe, _ = self.do_batch(img1s, img2s, labels, masks, _TAU=_TAUs, resize=resize)

                # calculate the metric for kitti dataset evaluation
                if return_type is not 'epe':
                    eps = 1e-8
                    epe = ((Norm(flows - labels) > 3) * ((Norm(flows - labels) / (Norm(labels) + eps)) > 0.05) * masks).sum(axis=0, exclude=True) / masks.sum(axis=0, exclude=True)
                
                batch_epe.append(epe)
            np_epes.append(np.concatenate([epe.asnumpy() for epe in batch_epe]))
        
        return np.mean(np.concatenate(np_epes, axis = 0), axis = 0)

    def predict(self, img1, img2, batch_size, resize = None):
        ''' predict the whole dataset
        '''
        size = len(img1)
        bs = batch_size
        
        for j in range(0, size, bs):
            batch_img1 = img1[j: j + bs]
            batch_img2 = img2[j: j + bs]

            batch_img1 = np.transpose(np.stack(batch_img1, axis=0), (0, 3, 1, 2))
            batch_img2 = np.transpose(np.stack(batch_img2, axis=0), (0, 3, 1, 2))

            batch_flow = []
            batch_disp_l_mask = []
        
            # Generate displacement maskss
            h, w =  batch_img1.shape[2:]
            _TAU = np.sqrt((h/5)**2 + (w/5)**2)
            _TAU = np.full(shape=(batch_img1.shape[0], 1, 1, 1), fill_value=_TAU, dtype=np.float32)

            ctx = self.ctx[ : min(len(batch_img1), len(self.ctx))]
            nd_img1, nd_img2, _TAU = map(lambda x : gluon.utils.split_and_load(x, ctx, even_split = False), 
                                         (batch_img1, batch_img2, _TAU))
            
            for img1s, img2s, _TAUs in zip(nd_img1, nd_img2, _TAU):
                img1s, img2s = img1s / 255.0, img2s / 255.0

                flows, _, disp_masks = self.do_batch(img1s, img2s, _TAU=_TAUs, resize=resize)
                
                batch_flow.append(flows)
                batch_disp_l_mask.append(disp_masks[-1])

            flow = np.concatenate([x.asnumpy() for x in batch_flow])
            flow = np.transpose(flow, (0, 2, 3, 1))
            flow = np.flip(flow, axis = -1)

            disp_l_mask = np.concatenate([x.asnumpy() for x in batch_disp_l_mask])
            disp_l_mask = np.transpose(disp_l_mask, (0, 2, 3, 1))

            for k in range(len(flow)):
                yield flow[k], disp_l_mask[k]