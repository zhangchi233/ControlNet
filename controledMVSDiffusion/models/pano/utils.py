import torch
import torch.nn.functional as F
from torch import nn
from ..modules.utils import get_x_2d, back_projection
#except:
#    from modules.utils import get_x_2d
from einops import rearrange



def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
def get_correspondences(depth, pose, K, x_2d):
    b, h, w = depth.shape
    x3d = back_projection(depth, pose, K, x_2d)
    x3d = rearrange(x3d, 'b h w c -> b c (h w)')
    x3d = K[:, :3, :3]@x3d
    x3d = rearrange(x3d, 'b c (h w) -> b h w c', h=h, w=w)
    x2d = x3d[..., :2]/(x3d[..., 2:3]+1e-6)

    mask = depth == 0
    x2d[mask] = -1000000
    x3d[mask] = -1000000

    return x2d, x3d

'''
def get_correspondences(R, K, img_h, img_w,T,depths):

    m = R.shape[1]
    correspondences=torch.zeros((R.shape[0], m, m, img_h, img_w, 2), device=R.device) # (b, m, m, h, w, 2) the last dim is corresponded x,y
    for i in range(m):  
        for j in range(m):

            R_right = R[:, j:j+1]
            K_right = K[:, j:j+1]
            l = R_right.shape[1]

            R_left = R[:, i:i+1].repeat(1, l, 1, 1)
            K_left = K[:, i:i+1].repeat(1, l, 1, 1)

            T_left = T[:, i:i+1].repeat(1, l, 1, 1)
            T_right = T[:, j:j+1].repeat(1, l, 1, 1)

            Depth_left = depths[:, i:i + 1]

            proj_left = torch.cat([R_left, T_left], dim=-1)
            proj_right = torch.cat([R_right, T_right], dim=-1)
            # append 0,0,0,1 to proj_left and proj_right
            proj_left = torch.cat([proj_left, torch.tensor([0, 0, 0, 1],
                                                           device=R.device).repeat(proj_left.shape[0],1,1,1)],dim = -2)
            proj_right = torch.cat([proj_right, torch.tensor([0, 0, 0, 1],
                                                                device=R.device).repeat(proj_right.shape[0],1,1,1)],dim = -2)
            #print(proj_left.shape,proj_right.shape)
            xy_l = homo_warping(proj_right, proj_left, Depth_left,img_h,img_w)
            
            R_left = R_left.reshape(-1, 3, 3)
            R_right = R_right.reshape(-1, 3, 3)
            K_left = K_left.reshape(-1, 3, 3)
            K_right = K_right.reshape(-1, 3, 3)


            homo_l = (K_right@torch.inverse(R_right) @ R_left@torch.inverse(K_left))
            # what 's this step's meaning?

            xyz_l = torch.tensor(get_x_2d(img_h, img_w),
                                device=R.device)
            xyz_l = (xyz_l.reshape(-1, 3).T)[None].repeat(homo_l.shape[0], 1, 1)
            
            xyz_l = homo_l@xyz_l 
            

            xy_l = (xyz_l[:, :2]/xyz_l[:, 2:]).permute(0,
                                                    2, 1).reshape(-1, l, img_h, img_w, 2)
            
            correspondences[:,i,j]=xy_l
    return correspondences
'''

def get_key_value(key_value, xy_l, homo_r, ori_h, ori_w, ori_h_r, query_h,depth):
    
    b, c, h, w = key_value.shape
    query_scale = ori_h//query_h
    key_scale = ori_h_r//h

    xy_l = xy_l[:, query_scale//2::query_scale,
                query_scale//2::query_scale]/key_scale-0.5
    depth_r = depth[:, query_scale//2::query_scale,
                query_scale//2::query_scale]

    key_values = []
    rot = homo_r[:, :3, :3]  # [B,3,3]
    trans = homo_r[:, :3, 3:4]  # [B,3,1]

    xy_proj = []
    kernal_size=3
    for i in range(0-kernal_size//2, 1+kernal_size//2):
        for j in range(0-kernal_size//2, 1+kernal_size//2):
            xy_l_norm = xy_l.clone()
            xy_l_norm[..., 0] = xy_l_norm[..., 0] + i # x
            xy_l_norm[..., 1] = xy_l_norm[..., 1] + j # y
            xy_l_rescale = (xy_l_norm+0.5)*key_scale

            xy_proj.append(xy_l_rescale) # the x,y coordinate

            xy_l_norm[..., 0] = xy_l_norm[..., 0]/(w-1)*2-1 # convert to pixel label
            xy_l_norm[..., 1] = xy_l_norm[..., 1]/(h-1)*2-1
            xy_l_norm = xy_l_norm.to(key_value.dtype)
            _key_value = F.grid_sample(
                key_value, xy_l_norm, align_corners=True)
            key_values.append(_key_value)

    xy_proj = torch.stack(xy_proj, dim=1)
    mask = (xy_proj[..., 0] > 0)*(xy_proj[..., 0] < ori_w) * \
        (xy_proj[..., 1] > 0)*(xy_proj[..., 1] < ori_h)
    depth = depth.unsqueeze(1)
    depths = []
    for proj in range(xy_proj.shape[1]):
        dep = F.grid_sample(depth, xy_proj[:, proj], align_corners=True)
        depths.append(dep)
    depth = torch.stack(depths, dim=1)

    xy_proj_back = torch.cat([xy_proj, torch.ones(
        *xy_proj.shape[:-1], 1, device=xy_proj.device)], dim=-1)
    xy_proj_back = rearrange(xy_proj_back, 'b n h w c -> b c (n h w)') # coordinate of the key_value

    depth = rearrange(depth, 'b n c h w -> b c (n h w)')
    xy_proj_back = rot@xy_proj_back
    xy_proj_back = xy_proj_back*depth
    xy_proj_back = xy_proj_back+trans
    
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


def get_query_value(query, key_value, xy_l, homo_r, img_h_l, img_w_l, depths_right, img_h_r=None, img_w_r=None):
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
        _key_value, _xy, _mask = get_key_value(key_value[:, i], xy_l[:, i], homo_r[:, i],
                                               img_h_l, img_w_l, img_w_r, q_h,depths_right[:,i])

        key_values.append(_key_value)
        xys.append(_xy)
        masks.append(_mask)

    key_value = torch.cat(key_values, dim=1)
    xy = torch.cat(xys, dim=1)
    mask = torch.cat(masks, dim=1)

    return query, key_value, xy, mask
def homo_warping(src_proj, ref_proj, depth_values,h,w):
    # src_fea: [B, C, H, W]
    # src_proj: [B,1, 4, 4]
    # ref_proj: [B,1, 4, 4]
    # depth_values: [B,1,h,w]
    # out: [B, C, Ndepth, H, W]
    batch = src_proj.shape[0]

    height, width = h,w

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:,0, :3, :3]  # [B,3,3]
        trans = proj[:,0, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_proj.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_proj.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        depth_values = depth_values.view(batch,height*width).unsqueeze(1) #[b,h*w]

        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]

        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
       # print(xyz.shape)
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz * depth_values # [B, 3, H*W]
       # print(rot_depth_xyz.shape)
        proj_xyz = rot_depth_xyz + trans # [B, 3, H*W]
        proj_xy = proj_xyz[:, :2, :] / proj_xyz[:, 2:3, :]  # [B, 2, H*W]
        #proj_x_normalized = proj_xy[:, 0, :] / ((width - 1) / 2) - 1
        #proj_y_normalized = proj_xy[:, 1, :] / ((height - 1) / 2) - 1
        #proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=2)  # [B, H*W, 2]
        #print(proj_xy.shape)
        grid = proj_xy.transpose(1, 2).view(batch, height, width, 2)  # [B, H, W, 2]
        #print(grid.shape)
    #warped_src_fea = F.grid_sample(src_fea, grid.view(batch, height, width, 2), mode='bilinear',
    #                              padding_mode='zeros')
    #warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    #return warped_src_fea
    return grid

def find_corresponding_pixels(depth_img1, K1, R1, K2, R2):
    B, c, height, width = depth_img1.shape 
    corresponding_pixels = np.zeros((B,height, width, 2), dtype=np.int)

    # Inverse of intrinsic and rotation matrices
    K1_inv = np.linalg.inv(K1)
    R1_inv = np.linalg.inv(R1)

    for i in range(height):
        for j in range(width):
            # Step 1: Compute 3D point in first camera coordinate system
            Z1 = depth_img1[:,i, j]
            X1 = Z1 * (j * K1_inv[:,0, 0] + K1_inv[:,0, 2])
            Y1 = Z1 * (i * K1_inv[:,1, 1] + K1_inv[:,1, 2])
            P1 = np.array([X1, Y1, Z1])

            # Step 2: Transform 3D point to second camera coordinate system
            P2 = R2 @ (R1_inv @ P1)

            # Step 3: Project 3D point onto second image plane
            p2 = K2 @ P2
            i2, j2 = int(p2[1] / p2[2]), int(p2[0] / p2[2])

            corresponding_pixels[:,i, j] = [i2, j2]

    return corresponding_pixels
if __name__=="__main__":
        import numpy as np
        depth_img1 = np.zeros((3,200, 100))
        K1 = np.random.rand(3,3,3)
        K2 = np.random.rand(3,3,3)
        R1 = np.random.rand(3,3,4)
        R2 = np.random.rand(3,3,4)
        fine_corresponding_pixels = find_corresponding_pixels(depth_img1, K1, R1, K2, R2)