from calendar import c
import torch
import torch.nn as nn
from ..depth.modules import CPAttn
from einops import rearrange
from .utils import get_correspondences
from .utils import zero_module,conv_nd
import copy
class MultiViewBaseModel(nn.Module):
    def __init__(self, unet, config):
        super().__init__()

        self.unet = unet
        self.single_image_ft = config['single_image_ft']
        self.unet.train()
        self.overlap_filter=0.1
        trainiable = config['trainiable']
        if config['single_image_ft']:
            self.trainable_parameters = [(self.unet.parameters(), 0.01)]
        else:
            self.cp_blocks_encoder = nn.ModuleList()
            for i in range(len(self.unet.down_blocks)):
                self.cp_blocks_encoder.append(CPAttn(
                    self.unet.down_blocks[i].resnets[-1].out_channels, flag360=False))

            self.cp_blocks_mid = CPAttn(
                self.unet.mid_block.resnets[-1].out_channels, flag360=False)

            self.cp_blocks_decoder = nn.ModuleList()
            for i in range(len(self.unet.up_blocks)):
                self.cp_blocks_decoder.append(CPAttn(
                    self.unet.up_blocks[i].resnets[-1].out_channels, flag360=False))
            
            
            #for i, downsample_block in enumerate(self.unet.down_blocks):
            #    if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
            #        training_parameters+=list(downsample_block.resnets.parameters())
            #        training_parameters+=list(downsample_block.attentions.parameters())
                    
            #    else:
            #        if i<2:
            #            training_parameters+=list(downsample_block.resnets.parameters())
            if trainiable:
                self.trainable_control_encoder_down = copy.copy(self.cp_blocks_encoder)
                self.trainable_control_encoder_mid = copy.copy(self.cp_blocks_mid)
                zero_layers = []
                #conv_channels = self.unet.conv_in.in_channels
                channels = self.unet.conv_in.out_channels
                zero_layers.append(zero_module(conv_nd(2,channels,channels,3,padding=1)))
                for i, downsample_block in enumerate(self.unet.down_blocks):
                    channels = downsample_block.resnets[-1].out_channels
                    zero_layers.append(zero_module(conv_nd(2,channels,channels,3,padding=1)))
                mid_channels =  self.unet.mid_block.resnets[-1].out_channels
                zero_layers.append(zero_module(conv_nd(2,mid_channels,mid_channels,3,padding=1)))
            self.trainable_parameters = [(list(self.cp_blocks_mid.parameters()) + \
                    list(self.cp_blocks_decoder.parameters()) + \
                    list(self.cp_blocks_encoder.parameters()), 1.0),
                    (list(self.unet.parameters()),0.01
                    #list(self.unet.mid_block .parameters()) +  
                    #list(self.unet.up_blocks.parameters())
                    )
                    ]
            #self.trainable_parameters+=training_parameters 
    def build_zeroconv(self,channels_in,channels_out,dim,num):
        layer = zero_module(conv_nd(dim,channels_in,channels_out,num,paddint = 1))
        return layer
    def get_correspondences(self, cp_package):
        # compute correspondence, in MVDiffusion, we use the correspondence to compute the correspondence-aware attention

        poses = cp_package['R']
        K = cp_package['K']
        depths = cp_package['Depth']
        cp_package['poses'] = poses
        cp_package["depths"]=depths
        b, m, h, w = depths.shape

        correspondence = torch.zeros(b, m, m, h, w, 2, device=depths.device) # m is m views, h, w is the height and width of the image
        
        K = rearrange(K, 'b m h w -> (b m) h w')
        overlap_ratios=torch.zeros(b, m, m, device=depths.device)
        
        for i in range(m):
            pose_i = poses[:, i:i+1].repeat(1, m, 1, 1)
            depth_i = depths[:, i:i+1].repeat(1, m, 1, 1)
            pose_j = poses
            depth_i = rearrange(depth_i, 'b m h w -> (b m) h w')
            pose_j = rearrange(pose_j, 'b m h w -> (b m) h w')
            pose_i = rearrange(pose_i, 'b m h w -> (b m) h w')
            pose_rel = torch.inverse(pose_j)@pose_i

            point_ij, _ = get_correspondences(
                depth_i, pose_rel, K, None)  # bs, 2, hw
            point_ij = rearrange(point_ij, '(b m) h w c -> b m h w c', b=b)
            correspondence[:, i] = point_ij
            mask=(point_ij[:,:,:,:,0]>=0)&(point_ij[:,:,:,:,0]<w)&(point_ij[:,:,:,:,1]>=0)&(point_ij[:,:,:,:,1]<h)
            mask=rearrange(mask, 'b m h w -> b m (h w)')
            overlap_ratios[:,i]=mask.float().mean(dim=-1)
        for b_i in range(b):
            for i in range(m):
                for j in range(i+1,m):
                    overlap_ratios[b_i, i, j] = overlap_ratios[b_i, j, i]=min(overlap_ratios[b_i, i, j], overlap_ratios[b_i, j, i])
        overlap_mask=overlap_ratios>self.overlap_filter # filter image pairs that have too small overlaps
        cp_package['correspondence'] = correspondence
        cp_package['overlap_mask']=overlap_mask

    def forward(self, latents, timestep, prompt_embd, meta):
        K = meta['K']
        R = meta['R']
        T = meta['T']
        latents = latents.to(self.unet.dtype)
        depths = meta['Depth']
        
        
        b, m, c, h, w = latents.shape
        img_h, img_w = h*8, w*8



        self.get_correspondences(meta)

        # bs*m, 4, 64, 64
        hidden_states = rearrange(latents, 'b m c h w -> (b m) c h w')
        prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')

        # 1. process timesteps

        timestep = timestep.reshape(-1)
        t_emb = self.unet.time_proj(timestep)  # (bs, 320)
        t_emb = t_emb.to(self.unet.dtype)
        emb = self.unet.time_embedding(t_emb)  # (bs, 1280)
        hidden_states = hidden_states.to(self.unet.dtype)
        hidden_states = self.unet.conv_in(
            hidden_states)  # bs*m, 320, 64, 64

        # unet
        # a. downsample
        prompt_embd = prompt_embd.to(self.unet.dtype)
        down_block_res_samples = (hidden_states,)
        for i, downsample_block in enumerate(self.unet.down_blocks):
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for resnet, attn in zip(downsample_block.resnets, downsample_block.attentions):
                    hidden_states = resnet(hidden_states, emb)

                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample

                    down_block_res_samples += (hidden_states,)
            else:
                for resnet in downsample_block.resnets:
                    hidden_states = resnet(hidden_states, emb)
                    down_block_res_samples += (hidden_states,)
            if m > 1:
                hidden_states = self.cp_blocks_encoder[i](
                    hidden_states, (img_h, img_w), meta, m)

            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = hidden_states.to(self.unet.dtype)
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)

        # b. mid
        hidden_states = hidden_states.to(self.unet.dtype)
        hidden_states = self.unet.mid_block.resnets[0](
            hidden_states, emb)

        if m > 1:
            hidden_states = hidden_states.to(self.unet.dtype)
            hidden_states = self.cp_blocks_mid(
               hidden_states, (img_h, img_w), meta, m)

        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            hidden_states = hidden_states.to(self.unet.dtype)
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embd
            ).sample
            hidden_states = resnet(hidden_states, emb)

        h, w = hidden_states.shape[-2:]

        # c. upsample
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                hidden_states = hidden_states.to(self.unet.dtype)
                for resnet, attn in zip(upsample_block.resnets, upsample_block.attentions):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample
            else:
                for resnet in upsample_block.resnets:
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
            if m > 1:
                hidden_states = hidden_states.to(self.unet.dtype)
                hidden_states = self.cp_blocks_decoder[i](
                    hidden_states, (img_h, img_w), meta, m)

            if upsample_block.upsamplers is not None:
                hidden_states = hidden_states.to(self.unet.dtype)
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        # 4.post-process
        hidden_states = hidden_states.to(self.unet.dtype)
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)
        return sample
