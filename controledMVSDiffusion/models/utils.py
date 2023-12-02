import os
os.chdir("/openbayes/input/input0")
print(os.getcwd())
import sys
sys.path.append("/openbayes/input/input0")
from CasMVSNet_pl.models.mvsnet import CascadeMVSNet 
from CasMVSNet_pl.datasets import DTUDataset
from CasMVSNet_pl.datasets.utils import read_pfm
import cv2
import copy
import numpy as np
from PIL import Image
import torch
import cv2
import argparse
#from src.dataset import MP3Ddataset, Scannetdataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from CasMVSNet_pl.utils import load_ckpt
from CasMVSNet_pl.utils import * 
import yaml
from src.lightning_pano_gen import PanoGenerator
#from src.lightning_pano_outpaint import PanoOutpaintGenerator
import pytorch_lightning as pl
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models.pano.MVGenModel import MultiViewBaseModel
from einops import rearrange
import torch.nn.functional as F
from src.models.modules.utils import get_x_2d
import torch.nn as nn
import torchvision.transforms as T
from inplace_abn import ABN

from src.models.modules.resnet import BasicResNetBlock
from src.models.modules.transformer import BasicTransformerBlock, PosEmbedding
#from src.models.pano.utils import get_query_value

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

def visualize_depth(x, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_


def findhomography(src_img,target_img):
    # Read the source and target images


    # Perform feature detection and matching (using ORB as an example)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(src_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(target_img, None)

    # Use a matcher (e.g., BFMatcher) to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract corresponding points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Use RANSAC to estimate the transformation matrix
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the source image to the target image
    result_img = cv2.warpPerspective(src_img, M, (target_img.shape[1], target_img.shape[0]))
    return result_img, M
    # Now, 'result_img' contains the source image warped to align with the target image


def get_correspondences(meta, img_h, img_w):
    homographys = meta["homographys"]
    m = homographys.shape[1]

    correspondences=torch.zeros((homographys.shape[0], m, m, img_h, img_w, 2), device=homographys.device)
    for i in range(m):
        for j in range(m):
            homo_l = homographys[:,i,j].float()


            xyz_l = torch.tensor(get_x_2d(img_h, img_w),
                                device=homographys.device)
            l = xyz_l.shape[-1]
            xyz_l = (
                xyz_l.reshape(-1, 3).T)[None].repeat(homo_l.shape[0], 1, 1)


            xyz_l = homo_l@xyz_l.float()
            l = 1
            #print(homo_l.shape,xyz_l.shape)
            xy_l = (xyz_l[:, :2]/xyz_l[:, 2:]).permute(0,
                                                    2, 1).reshape(-1, l, img_h, img_w, 2)

            correspondences[:,i,j]=xy_l[:,0]
    return correspondences.to("cuda") 
def get_key_value(key_value, xy_l, homo_r, ori_h, ori_w, ori_h_r, query_h):

    b, c, h, w = key_value.shape
    query_scale = ori_h//query_h
    key_scale = ori_h_r//h

    xy_l = xy_l[:, query_scale//2::query_scale,
                query_scale//2::query_scale]/key_scale-0.5

    key_values = []

    xy_proj = []
    kernal_size=3
    for i in range(0-kernal_size//2, 1+kernal_size//2):
        for j in range(0-kernal_size//2, 1+kernal_size//2):
            xy_l_norm = xy_l.clone()
            xy_l_norm[..., 0] = xy_l_norm[..., 0] + i
            xy_l_norm[..., 1] = xy_l_norm[..., 1] + j
            xy_l_rescale = (xy_l_norm+0.5)*key_scale

            xy_proj.append(xy_l_rescale)

            xy_l_norm[..., 0] = xy_l_norm[..., 0]/(w-1)*2-1
            xy_l_norm[..., 1] = xy_l_norm[..., 1]/(h-1)*2-1
            # print(key_value.device,xy_l_norm.device)
            _key_value = F.grid_sample(
                key_value, xy_l_norm, align_corners=True)
            key_values.append(_key_value)

    xy_proj = torch.stack(xy_proj, dim=1)
    mask = (xy_proj[..., 0] > 0)*(xy_proj[..., 0] < ori_w) * \
        (xy_proj[..., 1] > 0)*(xy_proj[..., 1] < ori_h)

    xy_proj_back = torch.cat([xy_proj, torch.ones(
        *xy_proj.shape[:-1], 1, device=xy_proj.device)], dim=-1)
    xy_proj_back = rearrange(xy_proj_back, 'b n h w c -> b c (n h w)')
    xy_proj_back = homo_r@xy_proj_back

    xy_proj_back = rearrange(
        xy_proj_back, 'b c (n h w) -> b n h w c', h=h, w=w)
    xy_proj_back = xy_proj_back[..., :2]/xy_proj_back[..., 2:]

    xy = get_x_2d(ori_w, ori_h)[:, :, :2]
    xy = xy[query_scale//2::query_scale, query_scale//2::query_scale]
    xy = torch.tensor(xy, device=key_value.device).float()[
        None, None]

    xy_rel = (xy_proj_back-xy)/query_scale

    key_values = torch.stack(key_values, dim=1)

    return key_values, xy_rel, mask

def get_query_value(query, key_value, xy_l, homo_r, img_h_l, img_w_l, img_h_r=None, img_w_r=None):
    if img_h_r is None:
        img_h_r = img_h_l
        img_w_r = img_w_l

    b = query.shape[0]
    m = key_value.shape[1]

    key_values = []
    masks = []
    xys = []

    for i in range(m):
        _, _, q_h, q_w = query.shape
        #print("shape is:",key_value.shape,query.shape,homo_r.shape,xy_l.shape)
        _key_value, _xy, _mask = get_key_value(key_value[:, i], xy_l[:, i], homo_r[:, i],
                                               img_h_l, img_w_l, img_w_r, q_h)

        key_values.append(_key_value)
        xys.append(_xy)
        masks.append(_mask)

    key_value = torch.cat(key_values, dim=1)
    xy = torch.cat(xys, dim=1)
    mask = torch.cat(masks, dim=1)

    return query, key_value, xy, mask
class CPBlock(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.attn1 = CPAttn(dim, flag360=flag360)
        self.attn2 = CPAttn(dim, flag360=flag360)
        self.resnet = BasicResNetBlock(dim, dim, zero_init=True)

    def forward(self, x, correspondences, img_h, img_w, R, K, m):
        x = self.attn1(x, correspondences, img_h, img_w, R, K, m)
        x = self.attn2(x, correspondences, img_h, img_w, R, K, m)
        x = self.resnet(x)
        return x


class CPAttn(nn.Module):
    def __init__(self, dim, flag360=False):
        super().__init__()
        self.flag360 = flag360
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = PosEmbedding(2, dim//4)

    def forward(self, x, correspondences, img_h, img_w, R, K, m,meta):
        b, c, h, w = x.shape
        x = rearrange(x, '(b m) c h w -> b m c h w', m=m)
        outs = []


        for i in range(m):
            indexs = [(i-1+m) % m, (i+1) % m]

            xy_r=correspondences[:, i, indexs]
            xy_l=correspondences[:, indexs, i]

            x_left = x[:, i]
            x_right = x[:, indexs]

            #R_right = R[:, indexs]
            #K_right = K[:, indexs]

            #l = R_right.shape[1]

            #R_left = R[:, i:i+1].repeat(1, l, 1, 1) # 1, l, 3,3
            #K_left = K[:, i:i+1].repeat(1, l, 1, 1)

            #R_left = R_left.reshape(-1, 3, 3)
            #R_right = R_right.reshape(-1, 3, 3)
            #K_left = K_left.reshape(-1, 3, 3)
            #K_right = K_right.reshape(-1, 3, 3)

            #homo_r = (K_left@torch.inverse(R_left) @
            #          R_right@torch.inverse(K_right))
            homo_r = meta["homographys"][:,indexs,i]
            #homo_r = rearrange(homo_r, '(b l) h w -> b l h w', b=xy_r.shape[0])
            #print("homo is:",homo_r.shape,x_left.shape,x_right.shape)
            query, key_value, key_value_xy, mask = get_query_value(
                x_left, x_right, xy_l, homo_r, img_h, img_w)

            key_value_xy = rearrange(key_value_xy, 'b l h w c->(b h w) l c')
            key_value_pe = self.pe(key_value_xy)

            key_value = rearrange(
                key_value, 'b l c h w-> (b h w) l c')
            mask = rearrange(mask, 'b l h w -> (b h w) l')

            key_value = (key_value + key_value_pe)*mask[..., None]

            query = rearrange(query, 'b c h w->(b h w) c')[:, None]
            query_pe = self.pe(torch.zeros(
                query.shape[0], 1, 2, device=query.device))

            out = self.transformer(query, key_value, query_pe=query_pe)

            out = rearrange(out[:, 0], '(b h w) c -> b c h w', h=h, w=w)
            outs.append(out)
        out = torch.stack(outs, dim=1)

        out = rearrange(out, 'b m c h w -> (b m) c h w')

        return out
class DTU(DTUDataset):
    def __init__(self,len,n_views=3,device = "cuda", levels=3, depth_interval=2.65,
                 img_wh=None,**kwargs):
        super().__init__(**kwargs)
        self.len = len
        self.model = self.build_model()
        self.n_views = n_views
        self.levels = levels # FPN levels
        self.depth_interval = depth_interval
        self.define_transforms()
    def __len__(self):
        return len(self.metas)
    def add_shadow(self,image, shadow_intensity=0.7):
        # Read the image
        original_image = copy.deepcopy(image)


        # Create a blank image with the same size as the original image
        shadow_image = np.zeros_like(original_image)

        # Define the shadow color (you can adjust these values as needed)
        shadow_color = (50, 50, 50)

        # Add the shadow to the image
        shadow_image[:, :] = shadow_color
        cv2.addWeighted(shadow_image, shadow_intensity, original_image, 1 - shadow_intensity, 0, original_image)
        return original_image
    def build_model(self):
        import os
        
        model = CascadeMVSNet(n_depths=[8,32,48],
                      interval_ratios=[1.0,2.0,4.0],
                      num_groups=1,
                      norm_act=ABN).cuda()
        load_ckpt(model, '/openbayes/input/input0/CasMVSNet_pl/ckpts/_ckpt_epoch_10.ckpt')
        
        
        return model
    def define_transforms(self):
        if self.split == 'train': # you can add augmentation here
            self.transform= T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
            self.unpreprocess = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                            std=[1/0.229, 1/0.224, 1/0.225])
            self.transform2= T.Compose([
                                        T.GaussianBlur(51, sigma=(1.0,3.0)),
                                       
                                       ])
        else:
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
    def build_proj_mats(self):
        proj_mats = []
        Rs = []
        Ks = []
        for vid in range(49): # total 49 view ids
            if self.img_wh is None:
                proj_mat_filename = os.path.join(self.root_dir,
                                                 f'Cameras/train/{vid:08d}_cam.txt')
            else:
                proj_mat_filename = os.path.join(self.root_dir,
                                                 f'Cameras/{vid:08d}_cam.txt')
            intrinsics, extrinsics, depth_min = \
                self.read_cam_file(proj_mat_filename)
            if self.img_wh is not None: # resize the intrinsics to the coarsest level
                intrinsics[0] *= self.img_wh[0]/1600/4
                intrinsics[1] *= self.img_wh[1]/1200/4

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_ls = []
            Ks.append(intrinsics)
            Rs.append(extrinsics)
            for l in reversed(range(self.levels)):
                proj_mat_l = np.eye(4)
                proj_mat_l[:3, :4] = intrinsics @ extrinsics[:3, :4]
                intrinsics[:2] *= 2 # 1/4->1/2->1
                proj_mat_ls += [torch.FloatTensor(proj_mat_l)]
            # (self.levels, 4, 4) from fine to coarse
            proj_mat_ls = torch.stack(proj_mat_ls[::-1])
            proj_mats += [(proj_mat_ls, depth_min)]
        self.R = Rs
        self.K = Ks
        self.proj_mats = proj_mats

    def build_metas(self):
        self.metas = []
        with open(f'/openbayes/input/input0/CasMVSNet_pl/datasets/lists/dtu/{self.split}.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = [3] if self.img_wh else range(7)

        pair_file = "Cameras/pair.txt"
        for scan in self.scans:
            with open(os.path.join(self.root_dir, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    for light_idx in light_idxs:
                        self.metas += [(scan, light_idx, ref_view, src_views)]
    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32) # (1200, 1600)
        if self.img_wh is None:
            depth = cv2.resize(depth, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            depth_0 = depth[44:556, 80:720] # (512, 640)
        else:
            depth_0 = cv2.resize(depth, self.img_wh,
                                 interpolation=cv2.INTER_NEAREST)


        return depth_0
    def extract_boundary(self,img, percentage_threshold=0.3):


        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        intensity_range = np.max(img) - np.min(img)
        dynamic_threshold = np.min(img) + percentage_threshold * intensity_range
        # Apply Canny edge detector
        #edges = cv2.Canny(blurred, dynamic_threshold, np.min(img) + 5*percentage_threshold * intensity_range)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_img, dynamic_threshold, 255, cv2.THRESH_BINARY)

        # Create a three-channel image to represent the edges
        #edges_color = cv2.merge([edges, edges, edges])

        # Combine the original image with the edges
        #plt.imshow(edges)
        #plt.show()
        #result = cv2.bitwise_and(img, edges)
        return binary_image

    def read_mask(self, filename):
        mask = cv2.imread(filename, 0) # (1200, 1600)
        #mask = cv2.resize(mask,(1200,1600))
        if self.img_wh is None:
            mask = cv2.resize(mask, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST) # (600, 800)
            mask_0 = mask[44:556, 80:720] # (512, 640)
        else:
            mask_0 = cv2.resize(mask, self.img_wh,
                                interpolation=cv2.INTER_NEAREST)
        mask_1 = cv2.resize(mask_0, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)
        mask_2 = cv2.resize(mask_1, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_NEAREST)


        return mask_0
    def calculate_mask(self,depth,result,threhold = 0.2):
        unpreprocess = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                           std=[1/0.229, 1/0.224, 1/0.225])
        pred_depth = visualize_depth(result["depth_0"].squeeze(0))
        depth = visualize_depth(depth)
        loss = torch.abs(pred_depth-depth).mean(dim = 0)
        # threhold is the ratio of losses to be mask
        loss_hist = loss.flatten()
        # find the threhold of loss at given threhold ratio
        loss_threhold = loss_hist.sort()[0][int(loss_hist.numel()*threhold)]
        mask = loss>loss_threhold
        return mask
    def __getitem__(self,idx):

        sample = {}
        scan, light_idx, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        dark_masks =[]
        prompts = []
        dark_imgs = []
        proj_mats = [] # record proj mats between views
        Rs = []
        Ks = []
        
        proj_mats = []
        masks = []
        depths = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            if self.img_wh is None:
                img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png')
                
                mask_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}/depth_visual_{vid:04d}.png')
                depth_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}/depth_map_{vid:04d}.pfm')
                prompts_file = os.path.join(self.root_dir,
                                            f'prompts/{scan}_train_prompt/rect_{1:03d}_{3}_r5000.png.txt')

            else:
                img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}/rect_{vid+1:03d}_{light_idx}_r5000.png')

            img = Image.open(img_filename)
            img = np.array(img)

            #dark_img = self.add_shadow(np.array(img))
            #img_mask = self.extract_boundary(np.array(dark_img))
            
            #dark_masks.append(torch.tensor(img_mask))
            
            dark_img = self.transform2(self.transform(img))
            dark_imgs.append(dark_img)

            if self.img_wh is not None:
                img = img.resize(self.img_wh, Image.BILINEAR)
            #img = self.transform(img)
            imgs += [img]

            proj_mat_ls, depth_min = self.proj_mats[vid]
            R = self.R[vid][:3,:3]
            K = self.K[vid]
            Rs.append(R)
            Ks.append(K)
            with open(prompts_file,"r") as f:
                prompt = f.readline().strip()+" in van goth sytled"
                prompts.append(prompt)

            #masks.append(self.read_mask(mask_filename))
            if i == 0:  # reference view
                sample['init_depth_min'] = torch.FloatTensor([depth_min])
                if self.img_wh is None:
                    #sample['masks'] = self.read_mask(mask_filename)
                    sample['depths'] = self.read_depth(depth_filename)
                ref_proj_inv = torch.inverse(proj_mat_ls)
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]
        prompts = [prompts[0] for i in range(len(prompts))]
        #dark_masks = torch.stack(dark_masks)
        images = []
        homographys = []
        for i,img in enumerate(imgs):
            image = self.transform(img)
            #image = torch.tensor(img)
            images.append(image)
            homograph = []
            for j in range(len(dark_imgs)):
                src_img = np.array(img)
                target_img = np.array(imgs[j])

                image,H = findhomography(src_img,target_img)
                homograph.append(H)
            homographys.append(homograph)
        homographys = torch.tensor(homographys)
        sample["homographys"] = homographys

        dark_imgs = np.stack(dark_imgs)
        proj_mats = torch.stack(proj_mats)[:,:,:3].to(self.device)
        dark_imgs = torch.tensor(dark_imgs)
        pred_depth = self.model(dark_imgs.unsqueeze(0).to(self.device), proj_mats.unsqueeze(0).to(self.device),
                                torch.tensor([depth_min]).reshape(-1,1).cuda(),
                                torch.tensor([self.depth_interval]).reshape(-1,1).cuda())
        sample_depth = sample["depths"]
        mask = self.calculate_mask(sample_depth,pred_depth)
        images = torch.stack(images) # views, 3, H, W)
        sample["mask"] = torch.tensor(mask).float()
        sample["pred_depth"] = pred_depth
        
        sample["prompts"] = prompts
        #sample["mask"] = dark_masks
        sample["dark_imgs"] = dark_imgs
        sample['imgs'] = images
        sample["images"] = images
        sample['proj_mats'] = proj_mats
        sample['depth_interval'] = torch.FloatTensor([self.depth_interval])
        sample['scan_vid'] = (scan, ref_view)
        sample["R"] = torch.tensor(Rs)
        sample["K"] = torch.tensor(Ks)

        return sample


