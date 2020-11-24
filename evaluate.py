import os
import yaml
import path
import argparse
import mxnet as mx

import network.config
from network import get_pipeline
from reader import sintel, kitti

parser = argparse.ArgumentParser()

parser.add_argument('--batch', type=int, default=8, help='minibatch size of samples per device')
parser.add_argument('config', type=str, nargs='?', default='MaskFlownet_DAM.yaml')

parser.add_argument('-g', '--gpu_device', type=str, default='', help='Specify gpu device(s)')
parser.add_argument('-c', '--checkpoint', type=str, default=None, required=True, 
    help='model checkpoint to load; by default, the latest one.'
    'You can use checkpoint:steps to load to a specific steps')
parser.add_argument('-n', '--network', type=str, default='MaskFlownet')
parser.add_argument('-d', '--dataset', type=str, default='sintel')

parser.add_argument('--weighted', action='store_true')

def find_checkpoint(ckpt):
    steps = 0
    if ':' in ckpt:
        prefix, steps = ckpt.split(':')
    else:
        prefix = ckpt
        steps = None

    log_file, run_id = path.find_log(prefix)    
    if steps is None:
        checkpoint, steps = path.find_checkpoints(run_id)[-1]

    steps = int(steps)

    print('Restore from steps: ', steps)

    return checkpoint, steps

def validate(batch_size):
    validation_result = {}
    for dataset_name in validation_datasets:
        validation_result[dataset_name] = pipe.validate(*validation_datasets[dataset_name], batch_size)
        
    return validation_result

def main():
    args = parser.parse_args()
    ctx = [mx.cpu()] if args.gpu_device == '' else [mx.gpu(gpu_id) for gpu_id in map(int, args.gpu_device.split(','))]

    with open(os.path.join('network', 'config', args.config)) as f:
        config = network.config.Reader(yaml.load(f, Loader=yaml.FullLoader))

    checkpoint, steps = find_checkpoint(args.checkpoint)

    pipe = get_pipeline(args.network, 
                        weighting_factor=args.weighted,
                        ctx=ctx, 
                        config=config)
    
    pipe.load(checkpoint)

    checkpoint_name = os.path.basename(checkpoint).replace('.params', '')
    
    if args.dataset == 'sintel':
        sintel_dataset = sintel.list_data()
        
        div = 'training'
        for k, dataset in sintel_dataset[div].items():
            img1, img2, flow, mask = [[sintel.load(p) for p in data] for data in zip(*dataset)]
            val_epe = pipe.validate(img1, img2, flow, mask, batch_size=args.batch)
            
            print('Sintel.{}.{}:epe={}'.format(div, k, val_epe))
            
    elif args.dataset == 'kitti':
        kitti_version = '2015'
        
        dataset = kitti.read_dataset(editions = kitti_version, parts = 'mixed', resize=(1216, 384))
        val_epe = pipe.validate(dataset['image_0'], 
                                dataset['image_1'], 
                                dataset['flow'], 
                                dataset['occ'], 
                                batch_size=args.batch, 
                                return_type = 'epe')
        
        print('KITTI.{}:epe={}'.format(kitti_version, val_epe))

        val_epe = pipe.validate(dataset['image_0'], 
                                dataset['image_1'], 
                                dataset['flow'], 
                                dataset['occ'], 
                                batch_size=args.batch, 
                                return_type = 'kitti')
        
        print('KITTI.{}:F1={}'.format(kitti_version, val_epe))
        
if __name__ == '__main__':
    main()
