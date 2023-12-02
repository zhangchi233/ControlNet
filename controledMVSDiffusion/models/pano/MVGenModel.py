from calendar import c
from os import times
import time
import torch
import torch.nn as nn
from ..utils import CPAttn
from einops import rearrange
from ..utils import get_correspondences
from .utils import zero_module,conv_nd
import copy
class MultiViewBaseModel(nn.Module):
    def __init__(self, unet, config):
        super().__init__()

        self.unet = unet
        self.single_image_ft = config['single_image_ft']
        self.unet.train()
        self.overlap_filter=0.1
        trainiable = config['Trainable']
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
                self.trainalbe_convin = copy.copy(self.unet.conv_in)
                for parm in self.trainalbe_convin.parameters():
                    parm.requires_grad = True
                self.trainable_downsample = copy.copy(self.unet.down_blocks)
                for parm in self.trainable_downsample.parameters():
                    parm.requires_grad = True
                self.trainable_mid = copy.copy(self.unet.mid_block)
                for parm in self.trainable_mid.parameters():
                    parm.requires_grad = True
                self.trainable_control_encoder_down = copy.copy(self.cp_blocks_encoder)
                for parm in self.trainable_control_encoder_down.parameters():
                    parm.requires_grad = True
                

                self.trainable_control_encoder_mid = copy.copy(self.cp_blocks_mid)
                for parm in self.trainable_control_encoder_mid.parameters():
                    parm.requires_grad = True
                self.zero_layers = []
                #conv_channels = self.unet.conv_in.in_channels
                channels = self.unet.conv_in.out_channels
                self.zero_layers.append(zero_module(conv_nd(2,channels,channels,3,padding=1)))
                for i, downsample_block in enumerate(self.unet.down_blocks):
                    channels = downsample_block.resnets[-1].out_channels
                    self.zero_layers.append(zero_module(conv_nd(2,channels,channels,3,padding=1)))
                
                mid_channels =  self.unet.mid_block.resnets[-1].out_channels
                self.zero_layers.append(zero_module(conv_nd(2,mid_channels,mid_channels,3,padding=1)))
                self.zero_layers = nn.ModuleList(self.zero_layers)
                for parm in self.zero_layers.parameters():
                    parm.requires_grad = True
            self.trainable_parameters = [(list(self.cp_blocks_mid.parameters()) + \
                    list(self.trainable_control_encoder_down.parameters()) + \
                    list(self.trainable_control_encoder_mid.parameters()), 1.0),
                    (list(self.trainalbe_convin.parameters())+list(self.trainable_downsample.parameters())
                     + list(self.trainable_mid.parameters()),0.01),
                    (list(self.zero_layers.parameters()),0.01)]
            #self.trainable_parameters+=training_parameters 
    def build_zeroconv(self,channels_in,channels_out,dim,num):
        layer = zero_module(conv_nd(dim,channels_in,channels_out,num,paddint = 1))
        return layer
    
    def forward_trainable(self, latents, timestep, prompt_embd,
                           meta,trainable_res,m,img_h,img_w,correspondences):
        count = 1
        K = meta['K']
        R = meta['R']
        T = meta['T']
        for i,downsample_block in enumerate(self.trainable_downsample):
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for resnet, attn in zip(downsample_block.resnets, downsample_block.attentions):
                    latents = resnet(latents, timestep)

                    latents = attn(
                        latents, encoder_hidden_states=prompt_embd
                    ).sample
                    
                    #zero_conv = self.zero_layers[count+1]
                    #trainable_res.append(zero_conv(latents))
                    
            else:
                for resnet in downsample_block.resnets:
                    latents = resnet(latents, timestep)
                    
            if m > 1:
                latents = self.trainable_control_encoder_down[i](
                   latents, correspondences, img_h, img_w, R, K, m,meta)
            trainable_res.append(self.zero_layers[count](latents))
            count+=1
            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    
                    latents = downsample(latents)
                
                    
                
            
        if m > 1:
            latents = latents.to(self.unet.dtype)
            lstents = self.trainable_control_encoder_mid(
            latents, correspondences, img_h, img_w, R, K, m,meta)
        for attn, resnet in zip(self.trainable_mid.attentions, self.trainable_mid.resnets[1:]):
            latents = latents.to(self.unet.dtype)
            latents = attn(
            latents, encoder_hidden_states=prompt_embd).sample
            latents = resnet(latents, timestep)
        zero_conv = self.zero_layers[count]
        trainable_res.append(zero_conv(latents))
        return trainable_res
   
    def forward(self, latents, timestep, prompt_embd, meta):
        K = meta['K']
        R = meta['R']
        T = meta['T']
        latents = latents.to(self.unet.dtype)
        depths = meta['Depth']
        
        
        b, m, c, h, w = latents.shape
        img_h, img_w = h*8, w*8



        correspondences = get_correspondences(meta, img_h, img_w)

        # bs*m, 4, 64, 64
        hidden_states = rearrange(latents, 'b m c h w -> (b m) c h w')
        prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')

        # 1. process timesteps

        timestep = timestep.reshape(-1)
        t_emb = self.unet.time_proj(timestep)  # (bs, 320)
        t_emb = t_emb.to(self.unet.dtype)
        emb = self.unet.time_embedding(t_emb)  # (bs, 1280)
        hidden_states = hidden_states.to(self.unet.dtype)
        hidden_trainable = hidden_states
        
        hidden_trainable =  self.trainalbe_convin(hidden_states)
        zero_layer = self.zero_layers[0]
        hidden_trainable = zero_layer(hidden_trainable)
            
        hidden_states = self.unet.conv_in(
            hidden_states)  # bs*m, 320, 64, 64
        hidden_trainable = hidden_states+hidden_trainable


        # unet
        # a. downsample
        prompt_embd = prompt_embd.to(self.unet.dtype)
        down_block_res_samples = (hidden_states,)
        training_add_res = []
        training_add_res.append(hidden_trainable)
        training_add_res = self.forward_trainable(hidden_trainable, emb, prompt_embd,meta,training_add_res,m,img_h,img_w,correspondences)
        
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
                    hidden_states, correspondences, img_h, img_w, R, K, m,meta)

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
               hidden_states, correspondences, img_h, img_w, R, K, m,meta)

        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            hidden_states = hidden_states.to(self.unet.dtype)
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embd
            ).sample
            hidden_states = resnet(hidden_states, emb)
        hidden_trainable = training_add_res[-1]
        training_add_res = training_add_res[:-1]
        hidden_states+=hidden_trainable

        h, w = hidden_states.shape[-2:]

        # c. upsample
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]

            hidden_states = hidden_states.to(self.unet.dtype)
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
                    hidden_states, correspondences, img_h, img_w, R, K, m,meta)

            
            if upsample_block.upsamplers is not None:
                hidden_trainable = training_add_res[-1]
            
                training_add_res = training_add_res[:-1]
                hidden_states+=hidden_trainable
                
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)
                
                

        # 4.post-process
        hidden_states = hidden_states.to(self.unet.dtype)
        hidden_states += training_add_res[-1]
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        #hidden_states += training_add_res[-2]
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)
        return sample
