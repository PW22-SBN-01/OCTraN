'''
Date: July 29th 2023
Description:
This python script contains the modules which comprises the contrastive occupancy transformer (CO-TraN).
'''

# IMPORTS
from PIL import Image
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR

from transformers import AutoImageProcessor, DPTFeatureExtractor

from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
import torch
from torch import nn
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

class OCTraN(pl.LightningModule):
    '''
    Contrastive Occupancy Transformer for 3D Semantic Scene Completition
    '''
    def __init__(self,lr):
        super().__init__()
        self.lr = lr
        self.weight_decay = 0.1
        self.train_metrics = nn.BCELoss()
        self.val_metrics = nn.BCELoss()
        self.test_metrics = nn.BCELoss()

        # log hyperparameters
        self.save_hyperparameters()

        # Initalize pretrained DPT
        self.image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        
        # Initialize embedding and encoder layer
        self.encode = Encoder3D()

        # Trainable Downsizing Layer
        self.img_feat_out_channels = 3
        self.kernal_size = 25
        self.stride = 2
        self.downsize = nn.Sequential(
            nn.Conv2d(3,4,self.kernal_size,stride=self.stride),
            nn.MaxPool2d(self.kernal_size, stride=self.stride),
            nn.Conv2d(4,8,self.kernal_size,stride=self.stride),
            Rearrange('b c h w -> b (h w) c')
        )

        # Initalize Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=8, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Initialize DeConv for Upsample
        self.upsample = nn.Sequential( # I have no fucking clue how I determined kernel size
            nn.ConvTranspose3d(8, 4, kernel_size=(21, 21, 4), stride=2),
            nn.ConvTranspose3d(4, 2, kernel_size=(20, 20, 4), stride=2),
            nn.ConvTranspose3d(2, 1, kernel_size=(18, 18, 6), stride=2)
        )
    
    def forward(self, img, vox):
        '''Compute forward pass'''

        # Obtain image depth features
        with torch.no_grad():
            # Extract image features
            # im = self.image_processor(img)
            logging.info(img.shape)
            img_features = self.feature_extractor(images=img,return_tensors="pt")['pixel_values'].to(device)
        
        # Downsize img_features
        img_features = self.downsize(img_features)

        # Obtain embedding + positional encoding
        logging.info(vox.shape) # Shape = (batch_size, 32, 32, 4)
        logging.info(vox.dtype) # 
        vox_encoded = self.encode(vox)
        # img_feat_encoded = self.encode(img_features)

        logging.info(f"Image Depth Features: {img_features.shape}")  # Size = (batch_size, 729, emb_dim)
        logging.info(f"Voxel Embedding: {vox_encoded.shape}")  # Size = (batch_size, 16, 16, 2, emb_dim)

        # Invoke Decoder
        out = rearrange(vox_encoded, 'b h w z c -> b (h w z) c')
        out = self.decoder(out,img_features)
        out = rearrange(out, 'b (h w z) c -> b c h w z', h=16,w=16,z=2)

        logging.info(f"Decoder Output: {out.shape}")  # Size = (batch_size, emb_dim, 16, 16, 2)

        # Upsample
        out = self.upsample(out)

        logging.info(f"Upsample Output: {out.shape}")  # Size = (batch_size, 1, 256, 265, 32)

        return out

    def step(self, batch, step_type, metric):
        logging.info(batch.keys())
        bs = len(batch["img"])
        loss = 0
        out_dict = self(batch['img'],torch.stack(batch["CP_mega_matrices"],dim=0).to(torch.float32))
        
        logging.info(f"Output: {out_dict}")

        # Define LOSS


        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
        return [optimizer], [scheduler]