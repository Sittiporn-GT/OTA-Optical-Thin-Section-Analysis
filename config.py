import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

#from tensorboardX import SummaryWriter

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'plutonic'
C.dataset_path = osp.join(C.root_dir, '/content/drive/MyDrive/datasets', 'plutonic')
C.rgb_root_folder = osp.join(C.dataset_path, 'PPL')
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'
C.gt_transform = False
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, 'XPL')
C.x_format = '.png'
C.x_is_single_channel = False # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, 'train.txt')
C.eval_source  = osp.join(C.dataset_path, 'val.txt')
C.test_source  = osp.join(C.dataset_path, 'test.txt')
C.is_test = False
C.num_train_imgs = 1668
C.num_eval_imgs = 222
C.num_classes = 15
C.class_names =  ['Background','Biotite','Clinopyroxene','Hornblende','K-feldspar','Leucite','Muscovite','Olivine','Opaque','Orthopyroxene','Plagioclase','Quartz','Spinel',
    'Topaz','Tourmaline']

"""Image Config"""
C.background = 255
C.image_height = 512
C.image_width = 512
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b2' # Remember change the path below.
C.pretrained_model = C.root_dir + '/drive/MyDrive/CMX_pretrained_models/segformers/mit_b2.pth'
C.decoder = 'UPernet'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
# --- Batch size & LR scaling ---
# Original: batch_size=4, lr=4e-5 (baseline)
# Scaling rule used: linear LR scaling with batch size (new_lr = base_lr * new_bs / base_bs)
C.base_batch_size = 4
C.base_lr = 4e-5

C.batch_size = 16
C.lr = C.base_lr * (C.batch_size / C.base_batch_size)

# --- Optional gradient accumulation ---
# Set > 1 to simulate a larger effective batch size without increasing VRAM usage.
# effective_batch_size = C.batch_size * C.grad_accum_steps
# If you raise this, remember to also scale C.lr by C.grad_accum_steps (or re-derive
# effective batch and use the same linear scaling rule above).
C.grad_accum_steps = 1

C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.nepochs = 300
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 4
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] # [0.75, 1, 1.25] # 
C.eval_flip = True # True # 
C.eval_crop_size = [512, 512] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 50
C.checkpoint_step = 50

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.drive_log_root = '/content/drive/MyDrive/CMX_logs'
C.log_dir = osp.abspath(osp.join(C.drive_log_root, 'log_' + C.dataset_name + '_' + C.backbone))
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(f"nepochs: {config.nepochs}")
    print(f"batch_size: {config.batch_size} (base was {config.base_batch_size})")
    print(f"lr: {config.lr} (base was {config.base_lr})")
    print(f"grad_accum_steps: {config.grad_accum_steps} -> effective batch size: {config.batch_size * config.grad_accum_steps}")
    print(f"niters_per_epoch: {config.niters_per_epoch}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    # if args.tensorboard:
    #     open_tensorboard()