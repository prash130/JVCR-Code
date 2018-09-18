# Utility Methods used for Image Transformation SET 1

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.misc


def transf_pred(pred_coord, center, scale):
    lm_pred = transform_preds(pred_coord, center, scale, [256, 256], 256)

    lm_pred[:, 2] = -lm_pred[:, 2]

    z_mean = torch.mean(lm_pred[:, 2])
    lm_pred[:, 2] -= z_mean

    return lm_pred
  
def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img
  
def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def img_crop(image_tensor, center, scale):
    return crop(image_tensor, center, scale, [256, 256])

def crop(img, center, scale, res, rot=0):
    img = im_to_numpy(img)

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = im_to_torch(scipy.misc.imresize(new_img, res))
    return new_img

def boundingbox(target_np):
    bbox = [np.min(target_np[:,0]), np.min(target_np[:,1]), np.max(target_np[:,0]), np.max(target_np[:,1])]
    bbox = np.array(bbox)

    bbox[2:4] = bbox[2:4] - bbox[0:2]

    center = bbox[0:2] + bbox[2:4] / 2.
    scale = bbox[2] / 200.

    return center, scale, bbox

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def draw_labelvolume(vol, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian 
    # ~~
    vol = to_numpy(vol)
    img = img = np.zeros((vol.shape[1:]))
    #img = vol.shape[1:]

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
 #   print ul, br
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        print("SOMETHING WRONG XXXX")
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    #print img_x, img_y
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    # extend to z-axis
    if vol.shape[0] == vol.shape[1]:
        z_gauss = g[x0]
    else:
        z_gauss = np.exp(- ((x - x0) ** 2) / (2 * sigma ** 2))

    z = np.uint8(pt[2])
    for i in range(len(z_gauss)):
        z_idx = z-x0+i
        if z_idx < 0 or z_idx >= vol.shape[0]:
            continue
        else:
          #  print z_gauss[i], img.shape
            vol[z_idx] = z_gauss[i] * img

    return to_torch(vol)
  
  
  

def show_voxel(pred_heatmap3d, ax=None):

    if ax is None:
        ax = plt.subplot(111, projection='3d')

    view_angle = (-160, 30)
    ht_map = pred_heatmap3d[0]
    density = ht_map.flatten()
    density = np.clip(density, 0, 1)
    density /= density.sum()
    selected_pt = np.random.choice(range(len(density)), 10000, p=density)
    pt3d = np.unravel_index(selected_pt, ht_map.shape)
    density_map = ht_map[pt3d]

    ax.set_aspect('equal')
    ax.scatter(pt3d[0], pt3d[2], pt3d[1], c=density_map, s=2, marker='.', linewidths=0)
    set_axes_equal(ax)
    # ax.set_xlabel('d', fontsize=10)
    # ax.set_ylabel('w', fontsize=10)
    # ax.set_zlabel('h', fontsize=10)
    ax.view_init(*view_angle)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])

    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('', fontsize=10)
    ax.set_zlabel('', fontsize=10)


