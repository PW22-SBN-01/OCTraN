'''
Date: July 29th 2023
Description:
This python script contains the modules which comprises the contrastive occupancy transformer (CO-TraN).
'''

# IMPORTS
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR

from PIL import Image
import numpy as np
import math

from transformers import AutoImageProcessor, DPTFeatureExtractor
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from OCTraN.loss.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss
from OCTraN.loss.sscMetrics import SSCMetrics
from OCTraN.config.constants import learning_map

# LOGGING
from OCTraN.utils.colored_logging import log as logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CLASSES
class Encoder3D(nn.Module):
    '''
    3D encoder which also appends positional encodings.
    '''
    def __init__(self,patch_size=(2,2,2),emb_dim=8):
        super().__init__()
        p1,p2,p3=patch_size
        patch_dim = math.prod(patch_size)
        # Layers
        self.patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) (d p3) -> b h w d (p1 p2 p3)', p1=p1,p2=p2,p3=p3),
            # nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim), # Replace with 3D Conv (encodes spatial info)?
            nn.LayerNorm(emb_dim),
        )
        self.pos_encoding = Summer(PositionalEncoding3D(emb_dim))

    def forward(self,input):
        # 1. Create Patch Embeddings
        patches_emb = self.patch_embedding(input)
        # 2. Add Positional Encoding
        patches_emb = self.pos_encoding(patches_emb)
        return patches_emb

class SegmentationHead(nn.Module):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    Taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    """

    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU()

        self.conv_classes = nn.Conv3d(
            planes, nbr_classes, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x_in):

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)

        return x_in

class OCTraN(pl.LightningModule):
    '''
    Contrastive Occupancy Transformer for 3D Semantic Scene Completition
    '''
    def __init__(
            self,
            n_classes,
            class_names,
            embeding_dim,
            class_weights,
            project_scale,
            full_scene_size,
            n_relations=4,
            context_prior=True,
            fp_loss=True,
            project_res=[],
            frustum_size=4,
            relation_loss=False,
            CE_ssc_loss=True,
            geo_scal_loss=True,
            sem_scal_loss=True,
            lr=1e-4,
            weight_decay=1e-4,
        ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_classes=n_classes
        self.class_names=class_names
        self.project_scale=project_scale
        self.full_scene_size=full_scene_size
        self.n_relations=n_relations
        self.context_prior=context_prior
        self.fp_loss=fp_loss
        self.frustum_size=frustum_size
        self.project_res=project_res
        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)
        self.CE_ssc_loss=CE_ssc_loss
        self.sem_scal_loss=sem_scal_loss
        self.geo_scal_loss=geo_scal_loss
        self.relation_loss=relation_loss
        self.class_weights=class_weights
        self.val_output=None

        # log hyperparameters
        self.save_hyperparameters()

        # Initalize pretrained DPT
        self.image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        
        # Initialize embedding and encoder layer
        self.embed = Encoder3D(emb_dim=embeding_dim)

        # Trainable Downsizing Layer
        self.img_feat_out_channels = 3
        self.kernal_size = 25
        self.stride = 2
        self.downsize = nn.Sequential(
            nn.Conv2d(3,8,self.kernal_size,stride=self.stride),
            nn.MaxPool2d(self.kernal_size, stride=self.stride),
            nn.Conv2d(8,embeding_dim,self.kernal_size,stride=self.stride),
            Rearrange('b c h w -> b (h w) c')
        )

        # Initalize Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embeding_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Initalize Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embeding_dim, nhead=8, batch_first=True)
        self.decoder1 = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.decoder2 = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.decoder3 = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.decoder4 = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.decoder5 = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Initialize DeConv for Upsample
        self.upsample = nn.Sequential( # I have no fucking clue how I determined kernel size
            nn.ConvTranspose3d(embeding_dim, embeding_dim, kernel_size=(2, 2, 2), stride=2),
            nn.Upsample(scale_factor=4,mode='trilinear'),
            nn.ConvTranspose3d(embeding_dim, embeding_dim, kernel_size=(2, 2, 2), stride=2),
        )

        # Segmentation head
        layer_size = [math.prod((256,256,32)),embeding_dim]
        self.seg_head = nn.Sequential(
            nn.LayerNorm(layer_size),
            nn.Linear(embeding_dim, 16),
            nn.LayerNorm(16),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.Linear(8, 1),
        )
        self.seg_head = SegmentationHead(32, 1, self.n_classes, [1, 2, 3])
    
    def forward(self, img, vox):
        '''Compute forward pass'''

        # Obtain image depth features
        with torch.no_grad():
            # Extract image features
            # im = self.image_processor(img)
            logging.info(img.shape)
            img_features = self.feature_extractor(images=img,return_tensors="pt")['pixel_values'].to(device)
        
        logging.info(f"Image Features B4 Downsize: {img_features.shape}")  # Size = (batch_size, 729, emb_dim)

        # Downsize img_features
        img_features = self.downsize(img_features)

        # Obtain embedding + positional encoding
        logging.info(vox.shape) # Shape = (batch_size, 32, 32, 4)
        vox_encoded = self.embed(vox)
        out = rearrange(vox_encoded, 'b h w z c -> b (h w z) c')

        logging.info(f"Image Depth Features: {img_features.shape}")  # Size = (batch_size, 729, emb_dim)
        logging.info(f"Voxel Embedding: {vox_encoded.shape}")  # Size = (batch_size, 16, 16, 2, emb_dim)

        # Invoke Decoder
        x = torch.bernoulli(torch.rand(8,512,512).uniform_(0,1))
        mask = torch.where(x==1,x,float('-inf')).to(device)

        out = self.decoder1(out,img_features,tgt_mask=mask)
        out = self.decoder2(out,img_features,tgt_mask=mask)
        out = self.decoder4(out,out)
        out = self.decoder5(out,out)
        
        out = rearrange(out, 'b (h w z) c -> b c h w z', h=16,w=16,z=2)

        logging.info(f"Decoder Output: {out.shape}")  # Size = (batch_size, emb_dim, 16, 16, 2)

        # Upsample
        out = self.upsample(out)
        logging.info(f"Upsample Output: {out.shape}")  # Size = (batch_size, 1, 256, 265, 32)

        # Segmentation
        out = self.seg_head(out)
        logging.info(f"Segmented Output: {out.shape}")  # Size = 

        # x3d_up_l1 = out.squeeze().permute(1,2,3,0).reshape(-1, 32)
        # logging.info(f"Reshaped Output: {x3d_up_l1.shape}")  # Size = (batch_size, 1, 256, 265, 32)

        # ssc_logit_full = self.seg_head(x3d_up_l1)

        # out = ssc_logit_full.reshape(256, 256, 32, 1).permute(3,0,1,2).unsqueeze(0)
        # logging.info(f"Output: {out.shape}")  # Size = (batch_size, 1, 256, 265, 32)

        return out

    def step(self, batch, step_type, metric):
        logging.info(batch.keys())
        bs = len(batch["img"])
        loss = 0

        ssc_pred = self(batch['img'],torch.stack(batch["CP_mega_matrices"],dim=0).to(torch.float32))
        target = batch["target"]

        logging.info(f'Input Unique: {torch.stack(batch["CP_mega_matrices"],dim=0).to(torch.float32).unique()}')
        logging.info(f"Model Output: {ssc_pred}")
        logging.info(f"Ground Truth: {target.unique()}")

        # Define LOSS
        logging.info(self.class_weights)
        class_weight = self.class_weights.type_as(batch["img"])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log(
                step_type + "/loss_ssc",
                loss_ssc.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.sem_scal_loss:
            loss_sem_scal = sem_scal_loss(ssc_pred, target)
            loss += loss_sem_scal
            self.log(
                step_type + "/loss_sem_scal",
                loss_sem_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.geo_scal_loss:
            loss_geo_scal = geo_scal_loss(ssc_pred, target)
            loss += loss_geo_scal
            self.log(
                step_type + "/loss_geo_scal",
                loss_geo_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.fp_loss and step_type != "test":
            frustums_masks = torch.stack(batch["frustums_masks"])
            frustums_class_dists = torch.stack(
                batch["frustums_class_dists"]
            ).float()  # (bs, n_frustums, n_classes)
            n_frustums = frustums_class_dists.shape[1]

            pred_prob = F.softmax(ssc_pred, dim=1)
            batch_cnt = frustums_class_dists.sum(0)  # (n_frustums, n_classes)

            frustum_loss = 0
            frustum_nonempty = 0
            for frus in range(n_frustums):
                frustum_mask = frustums_masks[:, frus, :, :, :].unsqueeze(1).float()
                prob = frustum_mask * pred_prob  # bs, n_classes, H, W, D
                prob = prob.reshape(bs, self.n_classes, -1).permute(1, 0, 2)
                prob = prob.reshape(self.n_classes, -1)
                cum_prob = prob.sum(dim=1)  # n_classes

                total_cnt = torch.sum(batch_cnt[frus])
                total_prob = prob.sum()
                if total_prob > 0 and total_cnt > 0:
                    frustum_target_proportion = batch_cnt[frus] / total_cnt
                    cum_prob = cum_prob / total_prob  # n_classes
                    frustum_loss_i = KL_sep(cum_prob, frustum_target_proportion)
                    frustum_loss += frustum_loss_i
                    frustum_nonempty += 1
            frustum_loss = frustum_loss / frustum_nonempty
            loss += frustum_loss
            self.log(
                step_type + "/loss_frustums",
                frustum_loss.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)
        logging.info(f"Loss: {loss}")
        logging.info(f"Metrics: {metric.get_stats()}")
        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.val_output = self.step(batch, "val", self.val_metrics)

    def on_validation_epoch_end(self,out, batch, batch_idx):
        self.log("val/Output",out)

    def on_validation_epoch_end(self):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    "{}_SemIoU/{}".format(prefix, class_name),
                    stats["iou_ssc"][i],
                    sync_dist=True,
                )
            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics)

    def test_epoch_end(self, outputs):
        classes = self.class_names
        metric_list = [("test", self.test_metrics)]
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print(
                "Precision={:.4f}, Recall={:.4f}, IoU={:.4f}".format(
                    stats["precision"] * 100, stats["recall"] * 100, stats["iou"] * 100
                )
            )
            print("class IoU: {}, ".format(classes))
            print(
                " ".join(["{:.4f}, "] * len(classes)).format(
                    *(stats["iou_ssc"] * 100).tolist()
                )
            )
            print("mIoU={:.4f}".format(stats["iou_ssc_mean"] * 100))
            metric.reset()
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
        return [optimizer], [scheduler]