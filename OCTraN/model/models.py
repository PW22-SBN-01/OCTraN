import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, ResNet34_Weights


from OCTraN.model.regnet import regnetx_002, regnetx_004, regnetx_006, regnetx_040, regnetx_080
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, PositionalEncodingPermute2D

from OCTraN.model.perciever import Perceiver

project_root = os.getcwd()

class OccupancyGrid_FrozenBiFPN_Multihead_stereo_batched_highres(nn.Module):
#     def __init__(self, debug=False, regnet_const=regnetx_080, regnet_path='weights/RegNetX-8.0G-1045c007.pth'):
    def __init__(self, 
                 debug=False, 
#                  regnet_const=regnetx_002, 
#                  regnet_path=os.path.join(project_root, 'weights/RegNetX-200M-5e5535e1.pth',),
#                  bifpn_shapes=[56, 152, 368],
                 regnet_const=regnetx_080, 
                 regnet_path=os.path.join(project_root, 'weights/RegNetX-8.0G-1045c007.pth'),
                 bifpn_shapes=[240, 720, 1920],
                 num_attention_layers = 3,
                 batch_size = 2
                ):
        super(OccupancyGrid_FrozenBiFPN_Multihead_stereo_batched_highres, self,).__init__()
        
        self.debug = debug
        self.num_attention_layers = num_attention_layers
        
        self.resnet_fpn = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name='resnet50', 
            weights=ResNet50_Weights.DEFAULT,
            trainable_layers=5 # 0-5
        )

        self.flatten = nn.Flatten(start_dim=2)
        self.flatten_1 = nn.Flatten(start_dim=1)
#         self.params_size = 2**13 * 11**1 * 31**1 # 2793472
        self.params_size = 2**15 * 11**1 * 31**1 # 2793472

        self.fraction_size = 2**12
        self.num_heads = 2*3*3 # 2*3*3*11
        self.batch_size = batch_size
#         self.new_shape = (1, 2**5, 2**4 * 31, 2**4 * 11)
        self.new_shape = (1, 2**5, 2**6 * 11, 2**4 * 31)

        self.p_enc_3d_sum = Summer(PositionalEncoding1D(self.params_size))
        self.p_enc_3d = PositionalEncoding1D(self.params_size)
#         self.p_enc_3d = PositionalEncoding3D(1)

#         self.multihead_attn = nn.MultiheadAttention(
#             embed_dim = self.params_size // self.fraction_size, 
#             num_heads = self.params_size // self.fraction_size,
#             kdim = self.params_size // self.fraction_size -1, 
#         )
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim = self.params_size // self.fraction_size, # q dim
            num_heads = self.params_size // self.fraction_size, # num_heads
            kdim = self.params_size // self.fraction_size, # k dim
            vdim = self.params_size // self.fraction_size, # v dim
        )
        
        self.norm1 = nn.InstanceNorm3d(1)
        self.norm2 = nn.InstanceNorm3d(1)
        self.norm3 = nn.InstanceNorm3d(1)
        
        self.up1 = nn.ConvTranspose3d(1, 1, kernel_size=(5, 5, 5), stride=1)
        self.up2 = nn.ConvTranspose3d(1, 1, kernel_size=(5, 5, 5), stride=1)
        self.up3 = nn.ConvTranspose3d(1, 1, kernel_size=(5, 5, 5), stride=1)
        self.up4 = nn.ConvTranspose3d(1, 1, kernel_size=(5, 5, 5), stride=1)
        self.up5 = nn.ConvTranspose3d(1, 1, kernel_size=(5, 5, 5), stride=1)
        self.up6 = nn.ConvTranspose3d(1, 1, kernel_size=(5, 5, 5), stride=1)
        self.up7 = nn.ConvTranspose3d(1, 1, kernel_size=(5, 5, 5), stride=1)
        self.up8 = nn.ConvTranspose3d(1, 1, kernel_size=(5, 5, 5), stride=1)
        self.up9 = nn.ConvTranspose3d(1, 1, kernel_size=(5, 5, 5), stride=1)

        self.sigmoid = torch.nn.Sigmoid()

    def set_resnet_fnp_training(self, status=True):
        # Stops training of regnets if false
#         return
        for param in self.resnet_fpn.parameters():
            param.requires_grad = status
        
    def forward(self, x):
        image_02, image_03 = x
        batch_size = image_02.shape[0]
        
        if len(image_02.shape) == 3:
            image_02 = image_02.unsqueeze(0)
            image_03 = image_03.unsqueeze(0)

        if self.debug: print('image_03.shape', image_03.shape)
        if self.debug: print('-'*10)

        feats3 = self.resnet_fpn(image_02)
        feats4 = self.resnet_fpn(image_03)

        if self.debug: print("Feature shape")
        for f in feats3:
#             if self.debug: print(feats3[f].shape, '->', self.regmod3(feats3[f]).shape)
            if self.debug: print(feats3[f].shape, '->', self.flatten_1(feats3[f]).shape)
        if self.debug: print("-"*10)

#         keys_vals3_cat = torch.cat([self.regmod3(feats3[feature]) for feature in feats3], dim=1) #.unsqueeze(1)
#         keys_vals4_cat = torch.cat([self.regmod3(feats4[feature]) for feature in feats4], dim=1) #.unsqueeze(1)
#         if self.debug: print('keys_vals3_cat.shape', keys_vals3_cat.shape)
#         keys_vals3 = self.flatten(keys_vals3_cat)
#         keys_vals4 = self.flatten(keys_vals4_cat)

        keys_vals3 = torch.cat([self.flatten_1(feats3[feature]) for feature in feats3], dim=1) # .unsqueeze(1)
        keys_vals4 = torch.cat([self.flatten_1(feats4[feature]) for feature in feats4], dim=1) # .unsqueeze(1)
        

        if self.debug: print('keys_vals3.shape', keys_vals3.shape)
            
        
#         keys_vals_cat = torch.cat([keys_vals3, keys_vals4], dim=1)
#         if self.debug: print('keys_vals_cat.shape', keys_vals_cat.shape)
#         keys_vals_list = []
#         for i in range(keys_vals_cat.shape[0]):
#             keys_vals_list.append(keys_vals_cat[i].reshape((1, self.fraction_size, self.params_size // self.fraction_size)))
#         keys_vals = torch.cat(keys_vals_list, dim=0)
    
        keys_vals_tmp = torch.cat([keys_vals3, keys_vals4], dim=1)
        keys_vals = keys_vals_tmp.view(keys_vals_tmp.shape[0], 1, keys_vals_tmp.shape[1])
        
        if self.debug: print('keys_vals.shape', keys_vals.shape)
            
            
##########################################
#         positional_encoding = self.p_enc_3d(
#             torch.zeros((batch_size, net.new_shape[1], net.new_shape[2], net.new_shape[3], 1)).to(device=image_02.device)
#         )

#         positional_encoding_list = []
#         for i in range(positional_encoding.shape[0]):
#             positional_encoding_list.append(positional_encoding[i].reshape((1, net.fraction_size, net.params_size // net.fraction_size)))
#         positional_encoding_final = torch.cat(positional_encoding_list, dim=0)
##########################################

        positional_encoding = self.p_enc_3d(keys_vals)

        positional_encoding_list = []
        for i in range(positional_encoding.shape[0]):
            positional_encoding_list.append(positional_encoding[i].reshape((1, self.fraction_size, self.params_size // self.fraction_size)))
        positional_encoding_final = torch.cat(positional_encoding_list, dim=0)

        pe_keys_vals = self.p_enc_3d_sum(keys_vals)

        if self.debug: print('positional_encoding.shape', positional_encoding.shape)

        
        pe_keys_vals_list = []
        for i in range(pe_keys_vals.shape[0]):
            pe_keys_vals_list.append(pe_keys_vals[i].reshape((1, self.fraction_size, self.params_size // self.fraction_size)))
        pe_keys_vals_final = torch.cat(pe_keys_vals_list, dim=0)


        if self.debug: print('positional_encoding_final.shape', positional_encoding_final.shape)

        #     return
#         attn_out, attn_out_weights = self.multihead_attn(positional_encoding, keys_vals, keys_vals)
#         attn_out, attn_out_weights = self.multihead_attn(positional_encoding_final, pe_keys_vals_final, pe_keys_vals_final)
        attn_out, attn_out_weights = self.multihead_attn(pe_keys_vals_final, pe_keys_vals_final, pe_keys_vals_final)

        if self.debug: print('attn_out.shape, attn_out_weights.shape', attn_out.shape, attn_out_weights.shape)

        # attn_out.shape torch.Size([1, 4, 49104])
#         new_shape = (1, 2**4, 2*2*3*11, 3*31)
#         new_shape = (1, 2**4, 3*31, 2*2*3*11)

        attn_out_list = []
        for i in range(attn_out.shape[0]):
            attn_out_list.append(attn_out[i].reshape(self.new_shape))
        attn_out = torch.cat(attn_out_list, dim=0)

        if self.debug: print('attn_out.shape', attn_out.shape)

        if self.debug: print("-"*10)
        # a = attn_out.unsqueeze(0).permute(1,0,2,3,4)
        a = attn_out.unsqueeze(1)
#         a = self.norm1(a)
        if self.debug: print(a.shape)
        
        a = self.norm1(a)
        if self.debug: print(a.shape)
        a = self.up1(a)
        if self.debug: print(a.shape)
        a= self.up2(a)
        if self.debug: print(a.shape)
        a = self.up3(a)
        if self.debug: print(a.shape)
        a = self.up4(a)
        if self.debug: print(a.shape)
        a = self.up5(a)
        if self.debug: print(a.shape)
        a = self.up6(a)
        if self.debug: print(a.shape)
        a = self.up7(a)
        if self.debug: print(a.shape)
        a = self.up8(a)
        if self.debug: print(a.shape)
        a = self.up9(a)
        if self.debug: print(a.shape)
        
        y = self.sigmoid(a)
        return y


num_latents = 256
latent_dim = 512

upscale_size = 1
fourier_channels = 32 * 128
input_axis = 2
num_freq_bands = round(((float(fourier_channels) / input_axis) - 1.0)/2.0)


class OCTraN3D_Perceiver(nn.Module):
    def __init__(self, 
                debug=False,
                 
                input_channels = 2560,          # number of channels for each token of the input
                input_axis = input_axis,              # number of axis for input data (2 for images, 3 for video)
                num_freq_bands = num_freq_bands,          # number of freq bands, with original value (2 * K + 1)
                max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
                depth = 1,                   # depth of net. The shape of the final attention mechanism will be:
                                                #   depth * (cross attention -> self_per_cross_attn * self attention)
                num_latents = num_latents,           # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim = latent_dim,            # latent dimension
                cross_heads = 1,             # number of heads for cross attention. paper said 1
                latent_heads = 8,            # number of heads for latent self attention, 8
                cross_dim_head = 64,         # number of dimensions per cross attention head
                latent_dim_head = 64,        # number of dimensions per latent self attention head
                attn_dropout = 0.,
                ff_dropout = 0.,
                weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
                fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
                self_per_cross_attn = 2,      # number of self attention blocks per cross attention
                
                grid_shape = (2**3, 2**7, 2**7),
                
                upscale_size = upscale_size,
                latents_init = False
                ):
        super(OCTraN3D_Perceiver, self,).__init__()
        
        assert len(grid_shape) == 3, "grid_shape must be 3D"
        
        if type(latents_init) == type(False):
            latents_init = torch.randn(num_latents, latent_dim)
        
#         num_latents = grid_shape[0]*grid_shape[1]*grid_shape[2]
#         latent_dim = grid_shape[0]*grid_shape[1]*grid_shape[2]
        num_classes = grid_shape[0]*grid_shape[1]*grid_shape[2]
        self.grid_shape = grid_shape
        self.debug = debug
        
        
#         self.resnet_fpn = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
#             backbone_name='resnet50', 
#             weights=ResNet50_Weights.DEFAULT,
#             trainable_layers=0 # 0-5
#         )
        self.resnet_fpn = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name='resnet34', 
            weights=ResNet34_Weights.DEFAULT,
            trainable_layers=0 # 0-5
        )
        
        self.flatten_1 = nn.Flatten(start_dim=1)
        
        self.perciever = Perceiver(
            input_channels = input_channels,
            input_axis = input_axis,
            num_freq_bands = num_freq_bands,
            max_freq = max_freq,
            depth = depth,
            num_latents = num_latents,
            latent_dim = latent_dim,
            cross_heads = cross_heads,
            latent_heads = latent_heads,
            cross_dim_head = cross_dim_head,
            latent_dim_head = latent_dim_head,
            num_classes = num_classes,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            weight_tie_layers = weight_tie_layers,
            fourier_encode_data = fourier_encode_data,
            self_per_cross_attn = self_per_cross_attn,
            latents_init = latents_init,
            final_classifier_head = False
        )
        
        self.norm1 = nn.InstanceNorm3d(1)
        self.norm2 = nn.InstanceNorm3d(1)
        self.norm3 = nn.InstanceNorm3d(1)
        
        self.up1 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up2 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up3 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up4 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up5 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up6 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up7 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up8 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up9 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)

        self.sigmoid = torch.nn.Sigmoid()

    def set_resnet_fnp_training(self, status=True):
#         return
        # Stops training of regnets if false
        for param in self.resnet_fpn.parameters():
            param.requires_grad = status
            
    def forward(self, x):
        image_02, image_03 = x
        batch_size = image_02.shape[0]
        
        if len(image_02.shape) == 3:
            image_02 = image_02.unsqueeze(0)
            image_03 = image_03.unsqueeze(0)

        
        if self.debug: print('image_03.shape', image_03.shape)
            
        if self.debug: print('-'*10)
            
        feats3 = self.resnet_fpn(image_02)
        feats4 = self.resnet_fpn(image_03)
        
        feats3_0 = feats3['0']
        feats4_0 = feats4['0']

        perc_input = torch.cat([feats3_0, feats4_0], dim=1).permute(0, 2,3,1)
                
        if self.debug: print('perc_input.shape', perc_input.shape)
            
        perciever_res = self.perciever(perc_input)
        small_grid = perciever_res.view(perciever_res.shape[0], self.grid_shape[0], self.grid_shape[1], self.grid_shape[2])

        if self.debug: print('perciever_res.shape', perciever_res.shape)
        if self.debug: print('small_grid.shape', small_grid.shape)

        if self.debug: print("-"*10)
        # a = attn_out.unsqueeze(0).permute(1,0,2,3,4)
#         a = small_grid.unsqueeze(1)
        a = small_grid.unsqueeze(1)

        if self.debug: print(small_grid.shape)
        
        a = self.norm1(a)
        if self.debug: print(a.shape)
        a = self.up1(a)
        if self.debug: print(a.shape)
        a= self.up2(a)
        if self.debug: print(a.shape)
        a = self.up3(a)
        if self.debug: print(a.shape)
        a = self.up4(a)
        if self.debug: print(a.shape)
        a = self.up5(a)
        if self.debug: print(a.shape)
        a = self.up6(a)
        if self.debug: print(a.shape)
        a = self.up7(a)
        if self.debug: print(a.shape)
        a = self.up8(a)
        if self.debug: print(a.shape)
        a = self.up9(a)
        if self.debug: print(a.shape)
        
        y = self.sigmoid(a)
        return y
    
class OCTraN3D_Perceiver_Pure(nn.Module):
    def __init__(self, 
                debug=False,
                 
                input_channels = 6,          # number of channels for each token of the input
                input_axis = input_axis,              # number of axis for input data (2 for images, 3 for video)
                num_freq_bands = num_freq_bands,          # number of freq bands, with original value (2 * K + 1)
                max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
                depth = 1,                   # depth of net. The shape of the final attention mechanism will be:
                                                #   depth * (cross attention -> self_per_cross_attn * self attention)
                num_latents = num_latents,           # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim = latent_dim,            # latent dimension
                cross_heads = 1,             # number of heads for cross attention. paper said 1
                latent_heads = 8,            # number of heads for latent self attention, 8
                cross_dim_head = 64,         # number of dimensions per cross attention head
                latent_dim_head = 64,        # number of dimensions per latent self attention head
                attn_dropout = 0.,
                ff_dropout = 0.,
                weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
                fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
                self_per_cross_attn = 2,      # number of self attention blocks per cross attention
                
                grid_shape = (2**3, 2**7, 2**7),
                
                upscale_size = upscale_size,
                latents_init = False
                ):
        super(OCTraN3D_Perceiver_Pure, self,).__init__()
        
        assert len(grid_shape) == 3, "grid_shape must be 3D"
        
        if type(latents_init) == type(False):
            latents_init = torch.randn(num_latents, latent_dim)
        
        num_classes = grid_shape[0]*grid_shape[1]*grid_shape[2]
        self.grid_shape = grid_shape
        self.debug = debug
        
        
        self.flatten_1 = nn.Flatten(start_dim=1)
        
        self.perciever = Perceiver(
            input_channels = input_channels,
            input_axis = input_axis,
            num_freq_bands = num_freq_bands,
            max_freq = max_freq,
            depth = depth,
            num_latents = num_latents,
            latent_dim = latent_dim,
            cross_heads = cross_heads,
            latent_heads = latent_heads,
            cross_dim_head = cross_dim_head,
            latent_dim_head = latent_dim_head,
            num_classes = num_classes,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            weight_tie_layers = weight_tie_layers,
            fourier_encode_data = fourier_encode_data,
            self_per_cross_attn = self_per_cross_attn,
            latents_init = latents_init,
            final_classifier_head = False
        )
        
        self.norm1 = nn.InstanceNorm3d(1)
        self.norm2 = nn.InstanceNorm3d(1)
        self.norm3 = nn.InstanceNorm3d(1)
        
        self.up1 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up2 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up3 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up4 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up5 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up6 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up7 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up8 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up9 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)

        self.sigmoid = torch.nn.Sigmoid()

    def set_resnet_fnp_training(self, status=True):
        return
            
    def forward(self, x):
        image_02, image_03 = x
        batch_size = image_02.shape[0]
        
        if len(image_02.shape) == 3:
            image_02 = image_02.unsqueeze(0)
            image_03 = image_03.unsqueeze(0)

        
        if self.debug: print('image_03.shape', image_03.shape)
            
        if self.debug: print('-'*10)
            
        # feats3 = self.resnet_fpn(image_02)
        # feats4 = self.resnet_fpn(image_03)
        
        # feats3_0 = feats3['0']
        # feats4_0 = feats4['0']

        perc_input = torch.cat([image_02, image_03], dim=1).permute(0, 2,3,1)
                
        if self.debug: print('perc_input.shape', perc_input.shape)

        perciever_res = self.perciever(perc_input)
        small_grid = perciever_res.view(perciever_res.shape[0], self.grid_shape[0], self.grid_shape[1], self.grid_shape[2])

        if self.debug: print('perciever_res.shape', perciever_res.shape)
        if self.debug: print('small_grid.shape', small_grid.shape)

        if self.debug: print("-"*10)
        # a = attn_out.unsqueeze(0).permute(1,0,2,3,4)
#         a = small_grid.unsqueeze(1)
        a = small_grid.unsqueeze(1)

        if self.debug: print(small_grid.shape)
        
        a = self.norm1(a)
        if self.debug: print(a.shape)
        a = self.up1(a)
        if self.debug: print(a.shape)
        a= self.up2(a)
        if self.debug: print(a.shape)
        a = self.up3(a)
        if self.debug: print(a.shape)
        a = self.up4(a)
        if self.debug: print(a.shape)
        a = self.up5(a)
        if self.debug: print(a.shape)
        a = self.up6(a)
        if self.debug: print(a.shape)
        a = self.up7(a)
        if self.debug: print(a.shape)
        a = self.up8(a)
        if self.debug: print(a.shape)
        a = self.up9(a)
        if self.debug: print(a.shape)
        
        y = self.sigmoid(a)
        return y

class OCTraN3D_Perceiver_Chunked(nn.Module):
    def __init__(self, 
                debug=False,
                 
                input_channels = 512,          # number of channels for each token of the input
                input_axis = input_axis,              # number of axis for input data (2 for images, 3 for video)
                num_freq_bands = num_freq_bands,          # number of freq bands, with original value (2 * K + 1)
                max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
                depth = 1,                   # depth of net. The shape of the final attention mechanism will be:
                                                #   depth * (cross attention -> self_per_cross_attn * self attention)
                num_latents = num_latents,           # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim = latent_dim,            # latent dimension
                cross_heads = 1,             # number of heads for cross attention. paper said 1
                latent_heads = 8,            # number of heads for latent self attention, 8
                cross_dim_head = 64,         # number of dimensions per cross attention head
                latent_dim_head = 64,        # number of dimensions per latent self attention head
                attn_dropout = 0.,
                ff_dropout = 0.,
                weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
                fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
                self_per_cross_attn = 2,      # number of self attention blocks per cross attention
                
                grid_shape = (2**3, 2**7, 2**7), # (height, width, depth)
                
                upscale_size = upscale_size,
                latents_init = False
                ):
        super(OCTraN3D_Perceiver_Chunked, self,).__init__()
        
        assert len(grid_shape) == 3, "grid_shape must be 3D"
        
        self.N_grid = 4
        self.N_feats = 5
        self.transp_2 = nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=2)
        self.transp_4 = nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=4)
        self.transp_8 = nn.ConvTranspose2d(256, 256, kernel_size=(8, 8), stride=8)
        self.transp_16 = nn.ConvTranspose2d(256, 256, kernel_size=(16, 16), stride=16)
        
        if type(latents_init) == type(False):
            latents_init = torch.randn(num_latents, latent_dim)
        
        num_classes = grid_shape[0]*grid_shape[1]*grid_shape[2]
        self.grid_shape = grid_shape
        self.debug = debug
        
        self.resnet_fpn = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name='resnet34', 
            weights=ResNet34_Weights.DEFAULT,
            trainable_layers=0 # 0-5
        )
        
        self.flatten_1 = nn.Flatten(start_dim=1)
        
        self.perciever = Perceiver(
            input_channels = input_channels,
            input_axis = input_axis,
            num_freq_bands = num_freq_bands,
            max_freq = max_freq,
            depth = depth,
            num_latents = num_latents,
            latent_dim = latent_dim,
            cross_heads = cross_heads,
            latent_heads = latent_heads,
            cross_dim_head = cross_dim_head,
            latent_dim_head = latent_dim_head,
            num_classes = num_classes,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            weight_tie_layers = weight_tie_layers,
            fourier_encode_data = fourier_encode_data,
            self_per_cross_attn = self_per_cross_attn,
            latents_init = latents_init,
            final_classifier_head = False
        )
        
        self.norm1 = nn.InstanceNorm3d(1)
        self.norm2 = nn.InstanceNorm3d(1)
        self.norm3 = nn.InstanceNorm3d(1)
        
        self.up1 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up2 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up3 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up4 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up5 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up6 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up7 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up8 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up9 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)

        self.sigmoid = torch.nn.Sigmoid()

        
    def get_chunks(self, feats3):

        feats3_chunk = {}
        feats3_chunkwise = {}

        for feats_key in feats3:
            feat_j = feats3[feats_key]
            wj = feat_j.shape[3]
            Fwj = round(wj / float(self.N_grid))
            i = 0
            chunk_j = feat_j.chunk(self.N_grid, dim=3)
            feats3_chunk[feats_key] = chunk_j
            for ch_index in range(len(chunk_j)):
                feats3_chunkwise.setdefault(ch_index, [])

                if chunk_j[ch_index].shape[2]==2:
                    chunk = self.transp_16(chunk_j[ch_index])
                elif chunk_j[ch_index].shape[2]==4:
                    chunk = self.transp_8(chunk_j[ch_index])
                elif chunk_j[ch_index].shape[2]==8:
                    chunk = self.transp_4(chunk_j[ch_index])
                elif chunk_j[ch_index].shape[2]==16:
                    chunk = self.transp_2(chunk_j[ch_index])
                else:
                    chunk = chunk_j[ch_index]

                pad_diff = 48 - chunk.shape[3]
                if pad_diff>0:
                    if pad_diff%2==0: # even
                        chunk = F.pad(chunk, (pad_diff//2, pad_diff//2), "constant", 0)
                    else:
                        chunk = F.pad(chunk, (pad_diff//2, pad_diff//2+1), "constant", 0)

                feats3_chunkwise[ch_index].append(
                    chunk
                )

        chunk_list = []
        for ch_index in feats3_chunkwise:
            assert self.N_feats == len(feats3_chunkwise[ch_index])
            chunk_list.append(torch.concat(feats3_chunkwise[ch_index], dim=1))

        return chunk_list
    
    def set_resnet_fnp_training(self, status=True):
#         return
        # Stops training of regnets if false
        for param in self.resnet_fpn.parameters():
            param.requires_grad = status
            
    def forward(self, x):
        image_02, image_03 = x
        batch_size = image_02.shape[0]
        
        if len(image_02.shape) == 3:
            image_02 = image_02.unsqueeze(0)
            image_03 = image_03.unsqueeze(0)

        
        if self.debug: print('image_03.shape', image_03.shape)
            
        if self.debug: print('-'*10)
            
        feats3 = self.resnet_fpn(image_02)
        feats4 = self.resnet_fpn(image_03)
        
        feats3_0 = self.get_chunks(feats3)
        feats4_0 = self.get_chunks(feats4)
        
#         print('feats3_0[0].shape', feats3_0[0].shape)
        
        feats_chunked = []
        for chunk_index in range(self.N_grid):
            feats_chunked.append(
                torch.concat([feats3_0[chunk_index], feats4_0[chunk_index]], dim=1)
            )
        
        small_grid_list = []
        for chunk_index in range(self.N_grid):
            perc_input = feats_chunked[chunk_index].permute(0, 2,3,1)    
            if self.debug: print('perc_input.shape', perc_input.shape)
            
            perciever_res = self.perciever(perc_input)
            small_grid_i = perciever_res.view(
                perciever_res.shape[0], self.grid_shape[0], self.grid_shape[1], self.grid_shape[2]
            )

            if self.debug: print('perciever_res.shape', perciever_res.shape)
            if self.debug: print('small_grid_i.shape', small_grid_i.shape)
                
            small_grid_list.append(small_grid_i)
            
        small_grid = torch.concat(small_grid_list, dim=3)
        if self.debug: print('small_grid.shape', small_grid.shape)


        if self.debug: print("-"*10)
        a = small_grid.unsqueeze(1)

        if self.debug: print(small_grid.shape)
        
        a = self.norm1(a)
        if self.debug: print(a.shape)
        a = self.up1(a)
        if self.debug: print(a.shape)
        a= self.up2(a)
        if self.debug: print(a.shape)
        a = self.up3(a)
        if self.debug: print(a.shape)
        a = self.up4(a)
        if self.debug: print(a.shape)
        a = self.up5(a)
        if self.debug: print(a.shape)
        a = self.up6(a)
        if self.debug: print(a.shape)
        a = self.up7(a)
        if self.debug: print(a.shape)
        a = self.up8(a)
        if self.debug: print(a.shape)
        a = self.up9(a)
        if self.debug: print(a.shape)
        
        y = self.sigmoid(a)
        return y

class OCTraN3D_Perceiver_Chunked_2(nn.Module):
    def __init__(self, 
                debug=False,
                 
                input_channels = 512,          # number of channels for each token of the input
                input_axis = input_axis,              # number of axis for input data (2 for images, 3 for video)
                num_freq_bands = num_freq_bands,          # number of freq bands, with original value (2 * K + 1)
                max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
                depth = 1,                   # depth of net. The shape of the final attention mechanism will be:
                                                #   depth * (cross attention -> self_per_cross_attn * self attention)
                num_latents = num_latents,           # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim = latent_dim,            # latent dimension
                cross_heads = 1,             # number of heads for cross attention. paper said 1
                latent_heads = 8,            # number of heads for latent self attention, 8
                cross_dim_head = 64,         # number of dimensions per cross attention head
                latent_dim_head = 64,        # number of dimensions per latent self attention head
                attn_dropout = 0.,
                ff_dropout = 0.,
                weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
                fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
                self_per_cross_attn = 2,      # number of self attention blocks per cross attention
                
                grid_shape = (2**3, 2**7, 2**7), # (height, width, depth)
                
                upscale_size = upscale_size,
                latents_init = False
                ):
        super(OCTraN3D_Perceiver_Chunked_2, self,).__init__()
        
        assert len(grid_shape) == 3, "grid_shape must be 3D"
        
        self.N_grid = 4
        self.N_feats = 5
        self.transp_2 = nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=2)
        self.transp_4 = nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=4)
        self.transp_8 = nn.ConvTranspose2d(256, 256, kernel_size=(8, 8), stride=8)
        self.transp_16 = nn.ConvTranspose2d(256, 256, kernel_size=(16, 16), stride=16)
        
        if type(latents_init) == type(False):
            latents_init = torch.randn(num_latents, latent_dim)
        
        num_classes = grid_shape[0]*grid_shape[1]*grid_shape[2]
        self.grid_shape = grid_shape
        self.debug = debug
        
        self.resnet_fpn = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name='resnet34', 
            weights=ResNet34_Weights.DEFAULT,
            trainable_layers=0 # 0-5
        )
        
        self.flatten_1 = nn.Flatten(start_dim=1)
        
        self.perciever = Perceiver(
            input_channels = input_channels,
            input_axis = input_axis,
            num_freq_bands = num_freq_bands,
            max_freq = max_freq,
            depth = depth,
            num_latents = num_latents,
            latent_dim = latent_dim,
            cross_heads = cross_heads,
            latent_heads = latent_heads,
            cross_dim_head = cross_dim_head,
            latent_dim_head = latent_dim_head,
            num_classes = num_classes,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            weight_tie_layers = weight_tie_layers,
            fourier_encode_data = fourier_encode_data,
            self_per_cross_attn = self_per_cross_attn,
            latents_init = latents_init,
            final_classifier_head = False
        )
        
        self.norm1 = nn.InstanceNorm3d(1)
        self.norm2 = nn.InstanceNorm3d(1)
        self.norm3 = nn.InstanceNorm3d(1)
        
        self.up1 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up2 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up3 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up4 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up5 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up6 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up7 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up8 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)
        self.up9 = nn.ConvTranspose3d(1, 1, kernel_size=(upscale_size, upscale_size, upscale_size), stride=1)

        self.sigmoid = torch.nn.Sigmoid()

        
    def get_chunks(self, feats3):

        feats3_chunk = {}
        feats3_chunkwise = {}

        for feats_key in feats3:
            feat_j = feats3[feats_key]
            wj = feat_j.shape[3]
            Fwj = round(wj / float(self.N_grid))
            i = 0
            chunk_j = feat_j.chunk(self.N_grid, dim=3)
            feats3_chunk[feats_key] = chunk_j
            for ch_index in range(len(chunk_j)):
                feats3_chunkwise.setdefault(ch_index, [])

                if chunk_j[ch_index].shape[2]==2:
                    chunk = self.transp_16(chunk_j[ch_index])
                elif chunk_j[ch_index].shape[2]==4:
                    chunk = self.transp_8(chunk_j[ch_index])
                elif chunk_j[ch_index].shape[2]==8:
                    chunk = self.transp_4(chunk_j[ch_index])
                elif chunk_j[ch_index].shape[2]==16:
                    chunk = self.transp_2(chunk_j[ch_index])
                else:
                    chunk = chunk_j[ch_index]

                pad_diff = 48 - chunk.shape[3]
                if pad_diff>0:
                    if pad_diff%2==0: # even
                        chunk = F.pad(chunk, (pad_diff//2, pad_diff//2), "constant", 0)
                    else:
                        chunk = F.pad(chunk, (pad_diff//2, pad_diff//2+1), "constant", 0)

                feats3_chunkwise[ch_index].append(
                    chunk
                )

        chunk_list = []
        for ch_index in feats3_chunkwise:
            assert self.N_feats == len(feats3_chunkwise[ch_index])
            chunk_list.append(torch.concat(feats3_chunkwise[ch_index], dim=1))

        return chunk_list
    
    def set_resnet_fnp_training(self, status=True):
#         return
        # Stops training of regnets if false
        for param in self.resnet_fpn.parameters():
            param.requires_grad = status
            
    def forward(self, x):
        image_02, image_03 = x
        batch_size = image_02.shape[0]
        
        if len(image_02.shape) == 3:
            image_02 = image_02.unsqueeze(0)
            image_03 = image_03.unsqueeze(0)

        
        if self.debug: print('image_03.shape', image_03.shape)
            
        if self.debug: print('-'*10)
            
        feats3 = self.resnet_fpn(image_02)
        feats4 = self.resnet_fpn(image_03)
        
        feats3_0 = self.get_chunks(feats3)
        feats4_0 = self.get_chunks(feats4)
        
#         print('feats3_0[0].shape', feats3_0[0].shape)
        
        feats_chunked = []
        for chunk_index in range(self.N_grid):
            feats_chunked.append(
                torch.concat([feats3_0[chunk_index], feats4_0[chunk_index]], dim=1)
            )
        
        small_grid_list = []
        for chunk_index in range(self.N_grid):
            perc_input = feats_chunked[chunk_index].permute(0, 2,3,1)    
            if self.debug: print('perc_input.shape', perc_input.shape)
            
            perciever_res = self.perciever(perc_input)
            small_grid_i = perciever_res.view(
                perciever_res.shape[0], self.grid_shape[0], self.grid_shape[1], self.grid_shape[2]
            )

            if self.debug: print('perciever_res.shape', perciever_res.shape)
            if self.debug: print('small_grid_i.shape', small_grid_i.shape)
                
            small_grid_list.append(small_grid_i)
            
        small_grid = torch.concat(small_grid_list, dim=2)
        if self.debug: print('small_grid.shape', small_grid.shape)


        if self.debug: print("-"*10)
        a = small_grid.unsqueeze(1)

        if self.debug: print(small_grid.shape)
        
        a = self.norm1(a)
        if self.debug: print(a.shape)
        a = self.up1(a)
        if self.debug: print(a.shape)
        a= self.up2(a)
        if self.debug: print(a.shape)
        a = self.up3(a)
        if self.debug: print(a.shape)
        a = self.up4(a)
        if self.debug: print(a.shape)
        a = self.up5(a)
        if self.debug: print(a.shape)
        a = self.up6(a)
        if self.debug: print(a.shape)
        a = self.up7(a)
        if self.debug: print(a.shape)
        a = self.up8(a)
        if self.debug: print(a.shape)
        a = self.up9(a)
        if self.debug: print(a.shape)
        
        y = self.sigmoid(a)
        return y
