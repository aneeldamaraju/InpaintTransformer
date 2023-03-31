from utils.mask_utils import *
import torch
import numpy as np
from torch.nn import functional as F

def load_data(img_and_pts, args):
    training_data = []
    for itr in range(len(img_and_pts)):
        if not args.sdf:
            training_data.append([torch.tensor(img_and_pts[itr][0], device=args.dev), torch.tensor(img_and_pts[itr][1], device=args.dev)])
        else:
            training_data.append(
                [torch.tensor(img_and_pts[itr][0], device=args.dev), torch.tensor(img_and_pts[itr][1], device=args.dev),
                 torch.tensor(img_and_pts[itr][2], device=args.dev)])

    return training_data


def get_random_xy(num=1, patch_size=64 / 256):
    coords = torch.zeros([num, 2])
    coords.uniform_(patch_size / 2, 1 - patch_size / 2)
    return (256 * coords.numpy()).astype('int')


def gen_high_res_img_patch(img, coords, psz=64, out_psz=8):
    '''
    The image is high resolution, not the output patch (out_pszxout_psz)
    '''
    x = coords[:, 0]
    y = coords[:, 1]
    ds_patches = np.zeros([coords.shape[0], out_psz, out_psz])
    for itr in range(coords.shape[0]):
        img_patch = img[int(y[itr] - psz / 2):int(y[itr] + psz / 2), int(x[itr] - psz / 2):int(x[itr] + psz / 2)]
        ds_patch = F.interpolate(torch.tensor(img_patch, dtype=torch.float32)[None, None, ...], [out_psz, out_psz],
                                 mode='area').squeeze()
        ds_patches[itr, ...] = ds_patch.cpu()
    return ds_patches


def get_unmasked_aligned_blob_points(img, pts, mask_corners, num_pts, args):
    cand_pts = []
    for pt in pts:
        if not pt_in_rect(pt, mask_corners):
            cand_pts.append(pt)
    coords = np.array(cand_pts)[np.random.choice(len(cand_pts), num_pts, replace=False).astype('int')]
    generated_patches = gen_high_res_img_patch(img, coords, psz=args.psz, out_psz=args.psz)
    # print(generated_patches.shape)
    generated_patches_flat = generated_patches.reshape(generated_patches.shape[0], -1).astype('float32')
    return torch.tensor(generated_patches_flat, device=args.dev), torch.tensor(coords, device=args.dev)


def gen_mask_rect(pts, H=256, W=256, thresh=.2, psz=64):
    pt = pts[np.random.choice(pts.shape[0]), :]
    w_pt = pt[0]
    h_pt = pt[1]
    # height and width of the mask
    w_mask = np.random.randint(int(thresh * W * .6), int(thresh * W))
    h_mask = np.random.randint(int(thresh * H * .6), int(thresh * H))

    rand_x = np.random.uniform(.3, .7)
    rand_y = np.random.uniform(.3, .7)
    mask_l = np.max([w_pt - int(w_mask * rand_x), int(psz / 2)])
    mask_r = np.min([w_pt + int(w_mask * (1 - rand_x)), W - int(psz / 2)])
    mask_b = np.max([h_pt - int(h_mask * rand_y), int(psz / 2)])
    mask_t = np.min([h_pt + int(h_mask * (1 - rand_y)), H - int(psz / 2)])

    x_pos, y_pos = np.meshgrid(np.linspace(0, H - 1, H), np.linspace(0, W - 1, W))

    left_wall = x_pos > mask_l
    right_wall = x_pos < mask_r
    top_wall = y_pos < mask_t
    bot_wall = y_pos > mask_b

    mask_rect = left_wall * right_wall * top_wall * bot_wall
    return mask_rect, [mask_l, mask_b, mask_r, mask_t]


def get_random_mask_points(img, mask_corners, num_pts, args):
    coords = torch.round(get_mask_xy(mask_corners, num_pts))
    generated_patches = gen_high_res_img_patch(img, coords, psz=1, out_psz=1)
    generated_patches_flat = generated_patches.reshape(generated_patches.shape[0], -1).astype('float32')
    return torch.tensor(generated_patches_flat, device=args.dev), torch.tensor(coords, device=args.dev)


def get_mask_xy(mask_corners, num=1):
    [l, b, r, t] = mask_corners
    coords = torch.zeros([num, 2])
    coords[:, 0].uniform_(l, r)
    coords[:, 1].uniform_(b, t)
    return coords