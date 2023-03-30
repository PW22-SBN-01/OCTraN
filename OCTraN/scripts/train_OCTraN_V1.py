import os
import sys
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python OccupancyNetwork/notebooks/train.py --name sweep_perc_resnet_jan_15 --sweep_json configs/sweep_config_random_10_IOU.json
# CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup python OccupancyNetwork/notebooks/train_OCTraN3D_Perceiver_ResNet_Chunked_2.py &
project_root = os.getcwd().split('OccupancyNetwork/notebooks')[0]

sys.path.append(os.path.join(project_root, 'OccupancyNetwork', 'model'))
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import json
import traceback
import time

import numpy as np
import random
import torch
# REPRODUCIBILITY
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

torch.multiprocessing.set_start_method('spawn')

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchviz import make_dot
from torch.nn import ConvTranspose3d
from torchvision import transforms
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights, ResNet34_Weights

import wandb

from tqdm import tqdm
# from tqdm.notebook import tqdm
# from tqdm import tqdm_notebook as tqdm
import logging

from pathlib import Path

from kitti_iterator import kitti_raw_iterator
from efficientnet_pytorch import EfficientNet



from PIL import Image


import cv2
from matplotlib import pyplot as plt

from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, PositionalEncodingPermute2D

from bifpn import BiFPN
from regnet import regnetx_002, regnetx_004, regnetx_006, regnetx_040, regnetx_080

from OCTraN.model.OCTraN3D_helper import *

from RegressionModel import RegressionModel
from MultiLayeredMultiheadAttention import MultiLayeredMultiheadAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'Using device {device}')

print('torch.__version__', torch.__version__) # 1.13.0
print('torch.version.cuda', torch.version.cuda) # => 11.7
print('torch.backends.cudnn.version()', torch.backends.cudnn.version()) # => 8500
print('torch.cuda.device_count()', torch.cuda.device_count())

###############################################

# wandb login
wandb.login()

###############################################

# Training Constants

# 164, 164, 44
# 128, 128, 8
grid_scale = (3.5, 2.5, 2.0)
grid_size = (512/grid_scale[0], 128/grid_scale[1], 8/grid_scale[2])
# grid_size = (740/grid_scale[0], 532/grid_scale[1], 68/grid_scale[2])

# grid_sigma = 1.0
grid_sigma = None
grid_gaus_n = 1

ground_removal=True

val_percent = 0.1
batch_size = 2

# kitti_raw_path = os.path.join(os.path.expanduser("~"), "kitti", "raw")
kitti_raw_path = "/home/shared/Kitti"
# kitti_raw_path = os.path.join(os.path.expanduser("~"), "Datasets", "kitti", "raw")

enable_test_image = False
enable_test_smoothness_loss = False
# image_scale_x = 256.0 / 291.0
# image_scale_y = 1024.0 / 1200.0
image_scale_x = 256.0 / 291.0 /2.0
image_scale_y = 1024.0 / 1200.0 /2.0

num_latents = 256
latent_dim = 512

upscale_size = 1

fourier_channels = 32 * 128
input_axis = 2
num_freq_bands = round(((float(fourier_channels) / input_axis) - 1.0)/2.0)

###############################################


###############################################

# Transformations and Dataset Definition
def total_transform(img, unsqueeze=False):
    tfms = transforms.Compose([
#         transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (round(image_scale_y*1200), round(image_scale_x*291)))
    img = Image.fromarray(img)
    if unsqueeze:
        img = tfms(img).unsqueeze(0)
    else:
        img = tfms(img)
    return img

def total_transform_grey(img, unsqueeze=False):
    tfms = transforms.Compose([
#         transforms.Resize((256, 256)), 
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = cv2.resize(img, (round(image_scale_y*1200), round(image_scale_x*291)))
    img = Image.fromarray(img)
    if unsqueeze:
        img = tfms(img).unsqueeze(0)
    else:
        img = tfms(img)
    return img

kitti_iter_0001 = kitti_raw_iterator.KittiRaw(
    kitti_raw_base_path=kitti_raw_path,
    date_folder="2011_09_26",
    sub_folder="2011_09_26_drive_0001_sync",
    transform={
        'image_00': total_transform,
        'image_01': total_transform,
        'image_02': total_transform,
        'image_03': total_transform,
        'occupancy_mask_2d': total_transform_grey,
        'occupancy_grid': transforms.Compose([
            transforms.ToTensor()
        ])
    },
    
    grid_size = grid_size,
    scale = grid_scale,
    sigma = grid_sigma,
    gaus_n= grid_gaus_n,
    ground_removal=ground_removal
)

# dataset = torch.utils.data.ConcatDataset(
#     get_kitti_raw(
#         kitti_raw_base_path=kitti_raw_path,
#         transform={
#             'image_00': total_transform,
#             'image_01': total_transform,
#             'image_02': total_transform,
#             'image_03': total_transform,
#             'occupancy_mask_2d': total_transform_grey,
#             'occupancy_grid': transforms.Compose([
#                 transforms.ToTensor()
#             ])
#         },
#         grid_size = grid_size,
#         scale = grid_scale,
#         sigma = grid_sigma,
#         gaus_n= grid_gaus_n,
#         ground_removal=ground_removal
#     )[:30]
# )

# total_size = len(dataset)
# total_use = int(round(total_size*0.02))
# total_discard = total_size - total_use

# print("Total number of frames", total_size)
# print("Using only", total_use, "frames")

# dataset, _ = random_split(dataset, [total_use, total_discard], generator=torch.Generator().manual_seed(0))

# # Split into train / validation partitions
# n_val = int(len(dataset) * val_percent)
# n_train = len(dataset) - n_val
# train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# print('len(train_set)', len(train_set))

###############################################

# Network definition

# from models import OccupancyGrid_FrozenBiFPN_Multihead_stereo_batched_highres as OCTraN3D
from models import OCTraN3D_Perceiver_Chunked_2 as OCTraN3D

###############################################

# Train

def train_net():
    
    # REPRODUCIBILITY

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


    # (Initialize logging)
    # experiment = wandb.init(project=nb_name, entity="pw22-sbn-01", resume='allow', anonymous='must')
    experiment = wandb.init(resume='allow', anonymous='must')

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    val_percent = wandb.config.val_percent
    save_checkpoint = wandb.config.save_checkpoint
    img_scale = wandb.config.img_scale
    amp = wandb.config.amp
    save_points = wandb.config.save_points
    alpha = wandb.config.alpha
    freeze_regnet_epoch = wandb.config.freeze_regnet_epoch
    apply_loss_mask_prob = wandb.config.apply_loss_mask_prob
    ground_removal = wandb.config.ground_removal
    weight_decay = wandb.config.weight_decay
    depth = wandb.config.depth
    cross_heads = wandb.config.cross_heads
    latent_heads = wandb.config.latent_heads
    cross_dim_head = wandb.config.cross_dim_head
    latent_dim_head = wandb.config.latent_dim_head
    self_per_cross_attn = wandb.config.self_per_cross_attn

    max_freq = wandb.config.max_freq

    net = OCTraN3D(
        debug=False,
        input_channels = 2560,          # number of channels for each token of the input
        input_axis = input_axis,              # number of axis for input data (2 for images, 3 for video)
        num_freq_bands = num_freq_bands,          # number of freq bands, with original value (2 * K + 1)
        max_freq = max_freq,              # maximum frequency, hyperparameter depending on how fine the data is
        depth = depth,                   # depth of net. The shape of the final attention mechanism will be:
                                        #   depth * (cross attention -> self_per_cross_attn * self attention)
        num_latents = num_latents,           # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = latent_dim,            # latent dimension
        cross_heads = cross_heads,             # number of heads for cross attention. paper said 1
        latent_heads = latent_heads,            # number of heads for latent self attention, 8
        cross_dim_head = cross_dim_head,         # number of dimensions per cross attention head
        latent_dim_head = latent_dim_head,        # number of dimensions per latent self attention head
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
        fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        self_per_cross_attn = self_per_cross_attn,      # number of self attention blocks per cross attention
        
        grid_shape = (2**3, 2**7, 2**7),
        
        upscale_size = upscale_size,
        latents_init = False
    )

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        print(f'Model loaded from {args.load}')

    net = net.to(device=device)

    print('net all params')
    count_params = sum([param.nelement() for param in net.parameters()])
    mem_params = sum([param.nelement()*param.element_size() for param in net.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in net.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print('mem', mem / 1024.0 / 1024.0, ' MB')
    print('count_params', count_params)

    print('net trainable params')
    count_params = sum([param.nelement() for param in net.parameters()])
    mem_params = sum([param.nelement()*param.element_size() for param in net.parameters() if param.requires_grad])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in net.buffers() if buf.requires_grad])
    mem = mem_params + mem_bufs # in bytes
    print('mem', mem / 1024.0 / 1024.0, ' MB')
    print('count_params', count_params)

    print('resnet_fpn all params')
    mem_params = sum([param.nelement()*param.element_size() for param in net.resnet_fpn.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in net.resnet_fpn.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print('mem', mem / 1024.0 / 1024.0, ' MB')

    print('resnet_fpn trainable params')
    mem_params = sum([param.nelement()*param.element_size() for param in net.resnet_fpn.parameters() if param.requires_grad])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in net.resnet_fpn.buffers() if buf.requires_grad])
    mem = mem_params + mem_bufs # in bytes
    print('mem', mem / 1024.0 / 1024.0, ' MB')

    # 1. Create dataset
    dataset = torch.utils.data.ConcatDataset(
        get_kitti_raw(
            kitti_raw_base_path=kitti_raw_path,
            transform={
                'image_00': total_transform,
                'image_01': total_transform,
                'image_02': total_transform,
                'image_03': total_transform,
                'occupancy_mask_2d': total_transform_grey,
                'occupancy_grid': transforms.Compose([
                    transforms.ToTensor()
                ])
            },
            grid_size = grid_size,
            scale = grid_scale,
            sigma = grid_sigma,
            gaus_n= grid_gaus_n,
            ground_removal=ground_removal
        )[:1]
    )
    
    total_size = len(dataset)
    total_use = int(round(total_size*1.0))
    total_discard = total_size - total_use

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    
    assert n_val > 0, 'Validation set is 0'
    assert n_train > 0, 'Train set is 0'
    
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    assert len(val_set) > 0, 'Validation set is 0'
    assert len(train_set) > 0, 'Train set is 0'
    
    print(f'''Starting training:
        epochs: {wandb.config.epochs}
        batch_size: {wandb.config.batch_size}
        learning_rate: {wandb.config.learning_rate}
        val_percent: {wandb.config.val_percent}
        save_checkpoint: {wandb.config.save_checkpoint}
        img_scale: {wandb.config.img_scale}
        amp: {wandb.config.amp}
        save_points: {wandb.config.save_points}
        alpha: {wandb.config.alpha}
        freeze_regnet_epoch: {wandb.config.freeze_regnet_epoch}
        apply_loss_mask_prob: {wandb.config.apply_loss_mask_prob}
        ground_removal: {wandb.config.ground_removal}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(
        net.parameters(), 
        lr=learning_rate, 
        betas=(0.9,0.999), 
        eps=1e-08, 
        weight_decay=weight_decay, 
        amsgrad=False,
    )
    
    net.set_resnet_fnp_training(True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize IOU score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    BCELoss_criterion = nn.BCELoss()
    global_step = 0
    
    alpha = torch.tensor(alpha).to(device=device, dtype=torch.float32)
    
    def criterion(masks_pred, true_masks, use_mask=False):
        if use_mask:
            consider_mask = true_masks > 0.5
            masks_pred_masked = masks_pred[consider_mask]
            true_masks_masked = true_masks[consider_mask]
            return (
                BCELoss_criterion(masks_pred_masked, true_masks_masked)
              )
        else:
            return (
                BCELoss_criterion(masks_pred, true_masks)
              )

    # 5. Begin training
    for epoch in range(1, epochs+1):
        if epoch > freeze_regnet_epoch:
            print("Freezing regnet weights")
#             net.set_resnet_fnp_training(False)

        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch_index in range(batch_size, len(train_set), batch_size):
                try:
                    batch = [[], [], [], [], []]
                    for sub_index in range(batch_index-batch_size, batch_index, 1):
                        new_batch = train_set[sub_index]
                        for sub_cat in range(len(batch)):
                            batch[sub_cat] += [new_batch[sub_cat].unsqueeze(0)]

                    for sub_cat in range(len(batch)):
                        batch[sub_cat] = torch.cat(batch[sub_cat], dim=0)

                    image_00, image_01, image_02, image_03, true_masks = batch[:5]

                    image_00 = image_00.to(device=device, dtype=torch.float32)
                    image_01 = image_01.to(device=device, dtype=torch.float32)
                    image_02 = image_02.to(device=device, dtype=torch.float32)
                    image_03 = image_03.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)

                    if epoch > freeze_regnet_epoch:
    #                     print("Freezing regnet weights")
                        net.set_resnet_fnp_training(False)
                        pass

                    with torch.cuda.amp.autocast(enabled=amp):
                        res = net([image_02, image_03])
                        masks_pred = res.permute((1,0,2,3,4)).squeeze(0)

    #                     print('masks_pred.shape', masks_pred.shape)
    #                     print('true_masks.shape', true_masks.shape)

#                         loss = criterion(masks_pred, true_masks, use_mask=epoch>apply_loss_mask_epoch)
#                         loss = criterion(masks_pred, true_masks, use_mask=epoch%2==0)
#                         loss = criterion(masks_pred, true_masks, use_mask=global_step%2==0)
                        loss = criterion(masks_pred, true_masks, use_mask=apply_loss_mask_prob>random.random())

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(image_00.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    experiment.log({
                        'train_loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    division_step = (n_train // (1 * batch_size))
                    if division_step >= 0:
                        if global_step % division_step == 0:
                            histograms = {}
                            for tag, value in net.named_parameters():
                                if type(value)!=type(None) and type(value.grad)!=type(None):
                                    tag = tag.replace('/', '.')
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                            occupancy_grid_pred = masks_pred[0].cpu().detach().numpy()

                            pc = kitti_iter_0001.transform_occupancy_grid_to_points(occupancy_grid_pred, threshold=0.54, skip=1)
                            rgb = np.ones_like(pc) * 255
                            rgb[:,0] = 0
                            rgb[:,1] = 0
                            pc_rgb = np.hstack([pc, rgb])

                            occupancy_grid_gt = true_masks[0].cpu().detach().numpy()
                            pc_gt = kitti_iter_0001.transform_occupancy_grid_to_points(occupancy_grid_gt, skip=1)
                            rgb = np.ones_like(pc_gt) * 255
                            rgb[:,1] = 0
                            rgb[:,2] = 0
                            pc_gt_rgb = np.hstack([pc_gt, rgb])

                            IOU, dice_score, IOU_lidar = evaluate(net, val_set, device, batch_size=batch_size, amp=amp)
                            print('IOU, dice_score, IOU_lidar: {}, {}, {}'.format(IOU, dice_score, IOU_lidar))

                            scheduler.step(IOU)

                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'IOU': IOU,
                                'IOU_lidar': IOU_lidar,
                                'dice_score': dice_score,
                                "point cloud pred": wandb.Object3D(
                                    {
                                        "type": "lidar/beta",
                                        "points": pc_rgb,
                                    }
                                ),
                                "point cloud gt": wandb.Object3D(
                                    {
                                        "type": "lidar/beta",
                                        "points": pc_gt_rgb,
                                    }
                                ),
                                'images': wandb.Image(image_00[0].cpu()),
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
    #                 global_step += 1
                except Exception as ex:
                    print(ex)
                    traceback.print_exc()
            try:
                test_image(net, kitti_iter_0001, train_set, plot=False)
            except Exception as ex:
                print(ex)
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                print(f'Checkpoint {epoch} saved!')

##############################################

def main(args):
    SWEEP_ID_PATH = os.path.expanduser('~/Documents/sweep_id.txt')
    with open(args.sweep_json, 'r') as sweep_json_file:
        sweep_config = json.load(sweep_json_file)
    
    # for i in range(100):
    # while True:
    
    # if os.path.isfile(SWEEP_ID_PATH):
    #     with open(SWEEP_ID_PATH) as sweep_id_file:
    #         sweep_id = sweep_id_file.read()
    #         print('sweep_id (from file)', sweep_id)
    # else:
    #     sweep_id = wandb.sweep(sweep_config, project=nb_name, entity="pw22-sbn-01")
    #     with open(SWEEP_ID_PATH, 'w') as sweep_id_file:
    #         sweep_id_file.write(sweep_id)
    #     print('sweep_id (generated)', sweep_id)

    sweep_id = wandb.sweep(sweep_config, project=nb_name, entity="pw22-sbn-01")
    print('sweep_id (generated)', sweep_id)

    wandb.agent(sweep_id, function=train_net)
    
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Training script for occupancy network')
    parser.add_argument('--name', default="sweep_perc_chunked_resnet_feb_11",
                        help='Name of the experiment')
    parser.add_argument('--load', default=False,
                        help='Path to checkpoint to load from; default: random weights')
    parser.add_argument('--sweep_json', default='configs/Feb_21/sweep_config_OccupancyGrid3D_Perceiver_resnet_Mar_19.json',
                        help='Path to checkpoint to sweep json')

    args = parser.parse_args()

    nb_name = args.name

    # Checkpoints dir setting
    dir_checkpoint = os.path.join('checkpoints', nb_name)
    os.makedirs(dir_checkpoint, exist_ok=True)
    dir_checkpoint = Path(dir_checkpoint)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    main(args)