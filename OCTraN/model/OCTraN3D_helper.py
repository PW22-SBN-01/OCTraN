import os
import sys

import time

import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchviz import make_dot
from torch.nn import ConvTranspose3d
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
from tqdm.notebook import tqdm
# from tqdm import tqdm_notebook as tqdm
import logging

from torch import Tensor
from pathlib import Path

from kitti_iterator import kitti_raw_iterator
from kitti_iterator.helper import depth_color

from efficientnet_pytorch import EfficientNet

import numpy as np

from PIL import Image
import random

import cv2
from matplotlib import pyplot as plt

from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, PositionalEncodingPermute2D

# from bifpn import BiFPN
# from regnet import regnetx_002, regnetx_004, regnetx_006, regnetx_040, regnetx_080

key_list = ['image_00', 'image_01', 'image_02', 'image_03', 'occupancy_grid', 'image_00_raw', 'image_01_raw', 'image_02_raw', 'image_03_raw', 'roi_00', 'roi_01', 'roi_02', 'roi_03', 'K_00', 'K_01', 'K_02', 'K_03', 'R_00', 'R_01', 'R_02', 'R_03', 'T_00', 'T_01', 'T_02', 'T_03', 'calib_cam_to_cam', 'calib_imu_to_velo', 'calib_velo_to_cam', 'velodyine_points', 'occupancy_mask_2d', 'velodyine_points_camera']
k2i = {} # key_to_index
for ind, key in enumerate(key_list):
    k2i[key] = ind

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def total_transform(img, unsqueeze=False):
    tfms = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    if unsqueeze:
        img = tfms(img).unsqueeze(0)
    else:
        img = tfms(img)
    return img

def total_transform_grey(img, unsqueeze=False):
    tfms = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.fromarray(img)
    if unsqueeze:
        img = tfms(img).unsqueeze(0)
    else:
        img = tfms(img)
    return img

class KittiRaw(kitti_raw_iterator.KittiRaw):
    
    def __getitem__(self, index):
        dat = super().__getitem__(index)
        all_dat = []
        for key in key_list:
            all_dat.append(dat[key])
        return all_dat
        image_00 = dat['image_00']
        image_01 = dat['image_01']
        image_02 = dat['image_02']
        image_03 = dat['image_03']
        occupancy_grid = dat['occupancy_grid']
        
        return (image_00, image_01, image_02, image_03, occupancy_grid)

def get_kitti_raw(**kwargs):
    kitti_raw_base_path=kwargs['kitti_raw_base_path']
    kitti_tree = kitti_raw_iterator.get_kitti_tree(kitti_raw_base_path)
    kitti_raw = []
    for date_folder in kitti_tree:
        for sub_folder in kitti_tree[date_folder]:
            if '_extract' in sub_folder:
                continue
            kitti_raw.append(
                KittiRaw(
                    # kitti_raw_base_path=kitti_raw_base_path,
                    date_folder=date_folder,
                    sub_folder=sub_folder,
                    **kwargs
                )
            )
    return kitti_raw

def evaluate(net, dataloader, device=device, threshold=0.5, batch_size=1, amp=False):
    net.eval()
    num_val_batches = len(dataloader)
    IOU = 0.0
    IOU_lidar = 0.0
    dice_score = 0.0

    print("Validation")
    
    assert len(dataloader) > 0, "Validation set has no elements"
    assert num_val_batches > 0, "num_val_batches set has no elements"
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image_00, image_01, image_02, image_03, true_masks = batch[:5]
        image_00 = image_00.to(device=device, dtype=torch.float32)
        image_01 = image_01.to(device=device, dtype=torch.float32)
        image_02 = image_02.to(device=device, dtype=torch.float32)
        image_03 = image_03.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=amp):
            res = net([image_02, image_03])
            mask_pred = res.permute((1,0,2,3,4)).squeeze(0).squeeze(0)
#             mask_pred = res.permute((1,0,4,2,3)).squeeze(0)
            
            pred_mask = mask_pred > threshold
            gt_mask = true_masks > threshold
            
            pred_mask = pred_mask.cpu().detach().numpy()
            gt_mask = gt_mask.cpu().detach().numpy()
            
            intersection = np.logical_and(gt_mask, pred_mask)
            union = np.logical_or(gt_mask, pred_mask)
            IOU_dat = np.sum(intersection) / np.sum(union)
            IOU += IOU_dat
            
            IOU_lidar_dat = np.sum(intersection) / np.sum(gt_mask)
            IOU_lidar += IOU_lidar_dat
            dice_score += 2 * np.sum(intersection) / (np.sum(gt_mask) + np.sum(pred_mask))
            
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        num_val_batches = 1
    
    IOU = IOU / num_val_batches * 100.0
    IOU_lidar = IOU_lidar / num_val_batches * 100.0
    dice_score = dice_score / num_val_batches
    return IOU, dice_score, IOU_lidar


def test_image(net, kitti_iter_0001, train_set, device=device, plot=True, i=-1, batch_size=1, skip=1):
    if i==-1:
        i = random.randint(0,len(kitti_iter_0001)-batch_size)
    data = train_set[i]
    image_00, image_01, image_02, image_03, true_masks = data[:5]
    image_00_raw = data[5]
    image_02_raw = data[7]
    velodyine_points = data[k2i['velodyine_points']]
    
    img_id = '_02'
    roi = data[k2i['roi'+img_id]]
    x, y, w, h = roi
    R_cam = data[k2i['R'+img_id]]
    T_cam = data[k2i['T'+img_id]]

    calib_cam_to_cam = data[k2i['calib_cam_to_cam']]
    calib_imu_to_velo = data[k2i['calib_imu_to_velo']]
    calib_velo_to_cam = data[k2i['calib_velo_to_cam']]

    P_rect = calib_cam_to_cam['P_rect' + img_id].reshape(3, 4)[:3,:3]
    K_img = data[k2i['K'+img_id]]
    
    image_00 = image_00.to(device=device, dtype=torch.float32)
    image_01 = image_01.to(device=device, dtype=torch.float32)
    image_02 = image_02.to(device=device, dtype=torch.float32)
    image_03 = image_03.to(device=device, dtype=torch.float32)
    true_masks = true_masks.to(device=device, dtype=torch.float32)
    
    res = net([image_02, image_03])

    occupancy_grid_pred_list = res.cpu().detach().permute((1,0,2,3,4)).squeeze(0)
    occupancy_grid_pred = occupancy_grid_pred_list[0]
    
    occupancy_grid_gt = true_masks
    
    print('occupancy_grid_pred.shape', occupancy_grid_pred.shape)
    print('occupancy_grid_gt.shape', occupancy_grid_gt.shape)
    
    occupancy_grid_pred = occupancy_grid_pred.permute((1,2,0))
    occupancy_grid_gt = occupancy_grid_gt.permute((1,2,0))
    
    occupancy_grid_pred = occupancy_grid_pred.cpu().detach().numpy()
    occupancy_grid_gt = occupancy_grid_gt.cpu().detach().numpy()

    print('occupancy_grid_pred.shape', occupancy_grid_pred.shape)
    print('occupancy_grid_gt.shape', occupancy_grid_gt.shape)
    
    assert occupancy_grid_pred.shape == occupancy_grid_gt.shape
    
    pc = kitti_iter_0001.transform_occupancy_grid_to_points(occupancy_grid_pred, threshold=0.54, skip=skip)
    gt_pc = kitti_iter_0001.transform_occupancy_grid_to_points(occupancy_grid_gt, threshold=0.5, skip=skip)
    
    print(occupancy_grid_pred.shape, '->', pc.shape)
    
    os.makedirs("pointcloud_outputs", exist_ok=True)
    np.save('pointcloud_outputs/model_pc.npy', pc)
    np.save('pointcloud_outputs/gt_pc.npy', gt_pc)
    print("Number of points", np.sum(occupancy_grid_pred > 0.5), pc.shape)
    print(image_02.shape)
    input_img = 255*image_02.cpu().permute(1,2,0).numpy().squeeze()
    input_img = input_img.astype(np.uint8)
    input_img = cv2.resize(input_img, (w, h))
#     input_img = cv2.flip(input_img, 1)
    if plot:
        plt.imshow(input_img)
        plt.show()
    
    
    image_points = kitti_iter_0001.transform_occupancy_grid_to_image_space(occupancy_grid_pred, roi, K_img, R_cam, T_cam, P_rect)
    # image_points = kitti_iter_0001.transform_points_to_image_space(pc, roi, K_img, R_cam, T_cam, P_rect)
    image_points = cv2.normalize(image_points - np.min(image_points.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    dilatation_size = 3
    dilation_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))
    image_points = cv2.dilate(image_points, element)
    image_points = cv2.applyColorMap(cv2.normalize(image_points - np.min(image_points.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
    
    image_points_pred = image_points.copy()

    if plot:
        plt.imshow(image_points)
        plt.show()
    
    image_points = kitti_iter_0001.transform_occupancy_grid_to_image_space(occupancy_grid_gt, roi, K_img, R_cam, T_cam, P_rect)
    # image_points = kitti_iter_0001.transform_points_to_image_space(velodyine_points, roi, K_img, R_cam, T_cam, P_rect)
    image_points = cv2.normalize(image_points - np.min(image_points.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    dilatation_size = 3
    dilation_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))
    image_points = cv2.dilate(image_points, element)
    image_points = cv2.applyColorMap(cv2.normalize(image_points - np.min(image_points.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
    
    image_points_gt = image_points.copy()

    if plot:
        plt.imshow(image_points)
        plt.show()
    
    input_img = cv2.resize(image_02_raw, (image_points.shape[1], image_points.shape[0]))
    print('input_img.shape', input_img.shape)
    print('image_points.shape', image_points.shape)
    image_points_overlay = cv2.addWeighted(image_points, 0.3, input_img, 0.5, 0.0)
    
    if plot:
        plt.imshow(image_points_overlay)
        plt.show()

    return occupancy_grid_pred, res


def test_image_no_permute(net, kitti_iter_0001, train_set, device=device, plot=True, i=-1, batch_size=1, skip=1):
    if i==-1:
        i = random.randint(0,len(kitti_iter_0001)-batch_size)
    data = train_set[i]
    image_00, image_01, image_02, image_03, true_masks = data[:5]
    image_00_raw = data[5]
    image_02_raw = data[7]
    velodyine_points = data[k2i['velodyine_points']]
    
    img_id = '_02'
    roi = data[k2i['roi'+img_id]]
    x, y, w, h = roi
    R_cam = data[k2i['R'+img_id]]
    T_cam = data[k2i['T'+img_id]]

    calib_cam_to_cam = data[k2i['calib_cam_to_cam']]
    calib_imu_to_velo = data[k2i['calib_imu_to_velo']]
    calib_velo_to_cam = data[k2i['calib_velo_to_cam']]

    P_rect = calib_cam_to_cam['P_rect' + img_id].reshape(3, 4)[:3,:3]
    K_img = data[k2i['K'+img_id]]
    
    image_00 = image_00.to(device=device, dtype=torch.float32)
    image_01 = image_01.to(device=device, dtype=torch.float32)
    image_02 = image_02.to(device=device, dtype=torch.float32)
    image_03 = image_03.to(device=device, dtype=torch.float32)
    true_masks = true_masks.to(device=device, dtype=torch.float32)
    
    res = net([image_02, image_03])

    occupancy_grid_pred_list = res.cpu().detach().squeeze(0)
    occupancy_grid_pred = occupancy_grid_pred_list[0]
    
    occupancy_grid_gt = true_masks
    
    print('occupancy_grid_pred.shape', occupancy_grid_pred.shape)
    print('occupancy_grid_gt.shape', occupancy_grid_gt.shape)
    
    occupancy_grid_pred = occupancy_grid_pred.permute((1,2,0))
    occupancy_grid_gt = occupancy_grid_gt.permute((1,2,0))
    
    occupancy_grid_pred = occupancy_grid_pred.cpu().detach().numpy()
    occupancy_grid_gt = occupancy_grid_gt.cpu().detach().numpy()

    print('occupancy_grid_pred.shape', occupancy_grid_pred.shape)
    print('occupancy_grid_gt.shape', occupancy_grid_gt.shape)
    
    assert occupancy_grid_pred.shape == occupancy_grid_gt.shape
    
    pc = kitti_iter_0001.transform_occupancy_grid_to_points(occupancy_grid_pred, threshold=0.54, skip=skip)
    gt_pc = kitti_iter_0001.transform_occupancy_grid_to_points(occupancy_grid_gt, threshold=0.5, skip=skip)
    
    print(occupancy_grid_pred.shape, '->', pc.shape)
    
    os.makedirs("pointcloud_outputs", exist_ok=True)
    np.save('pointcloud_outputs/model_pc.npy', pc)
    np.save('pointcloud_outputs/gt_pc.npy', gt_pc)
    print("Number of points", np.sum(occupancy_grid_pred > 0.5), pc.shape)
    print(image_02.shape)
#     input_img = 255*image_02.cpu().permute(1,2,0).numpy().squeeze()
    input_img = image_02.cpu().numpy().squeeze() * 255.0
#     input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    input_img = input_img.astype(np.uint8)
    input_img = cv2.resize(input_img, (w, h))
#     input_img = cv2.flip(input_img, 1)
    if plot:
        plt.imshow(input_img)
        plt.show()
    
    
    image_points = kitti_iter_0001.transform_occupancy_grid_to_image_space(occupancy_grid_pred, roi, K_img, R_cam, T_cam, P_rect)
    # image_points = kitti_iter_0001.transform_points_to_image_space(pc, roi, K_img, R_cam, T_cam, P_rect)
    image_points = cv2.normalize(image_points - np.min(image_points.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    dilatation_size = 3
    dilation_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))
    image_points = cv2.dilate(image_points, element)
    image_points = cv2.applyColorMap(cv2.normalize(image_points - np.min(image_points.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
    
    image_points_pred = image_points.copy()

    if plot:
        plt.imshow(image_points)
        plt.show()
    
    image_points = kitti_iter_0001.transform_occupancy_grid_to_image_space(occupancy_grid_gt, roi, K_img, R_cam, T_cam, P_rect)
    # image_points = kitti_iter_0001.transform_points_to_image_space(velodyine_points, roi, K_img, R_cam, T_cam, P_rect)
    image_points = cv2.normalize(image_points - np.min(image_points.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    dilatation_size = 3
    dilation_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))
    image_points = cv2.dilate(image_points, element)
    image_points = cv2.applyColorMap(cv2.normalize(image_points - np.min(image_points.flatten()), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
    
    image_points_gt = image_points.copy()

    if plot:
        plt.imshow(image_points)
        plt.show()
    
    input_img = cv2.resize(image_02_raw, (image_points.shape[1], image_points.shape[0]))
    print('input_img.shape', input_img.shape)
    print('image_points.shape', image_points.shape)
    image_points_overlay = cv2.addWeighted(image_points, 0.3, input_img, 0.5, 0.0)
    
    if plot:
        plt.imshow(image_points_overlay)
        plt.show()

    return occupancy_grid_pred, res

def smoothness_loss(masks_pred, sigma=1.0, n=1, reduction='mean', device=device):
    assert reduction in ('none', 'mean', 'sum')
    x = np.arange(-n,n+1,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-n,n+1,1)
    z = np.arange(-n,n+1,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel =  - 1.0 * np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel[n,n,n] = 1
    
    normalizing_constant = -(np.sum(kernel) - 1)
    kernel = kernel / float(normalizing_constant)
    kernel[n,n,n] = 1
    kernel = kernel * 0.5
    
    kernel = torch.tensor(kernel).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

    filtered = torch.nn.functional.conv3d(masks_pred, kernel, stride=1)
    filtered = torch.square(filtered)
    if reduction == 'mean':
        return torch.mean(filtered)
    elif reduction == 'sum':
        return torch.sum(filtered)
    return filtered