import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import scipy.misc
from PIL import Image
import numpy as np
import torchvision
import torch

import sys
import argparse
from time import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
import scipy.misc
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.misc

import torch.nn as nn
import torch.nn.functional as F


def get_transform(center, scale, res, rot=0):
    """
    General image utils functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def transform3d(pt, center, scale, res, z_res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)

    h = 200 * scale
    if invert:
        new_pt_z = (h/float(z_res)) * (pt[2] - float(z_res)/2)
    else:
        new_pt_z = (float(z_res)/h) * pt[2] + float(z_res)/2

    new_pt[2] = new_pt_z

    return new_pt[:3]



def creat_volume(pts, center, scale, out_res, depth_res, sigma=1, label_type='Gaussian'):
    nStack = len(depth_res)
    target = []
    for i in range(nStack):
        target_i = torch.zeros(depth_res[i], out_res, out_res)
        tpts = pts.clone()
        for j in range(tpts.size(0)):
            # if tpts[j, 2] > 0: # This is evil!!
            if tpts[j, 0] > 0:
                target_j = torch.zeros(depth_res[i], out_res, out_res)
                tpts[j, 0:3] = to_torch(transform3d(tpts[j, 0:3] + 1, center, scale, [out_res, out_res],
                                                    depth_res[i], rot=0))
                target_j = draw_labelvolume(target_j, tpts[j] - 1, sigma, type=label_type)
                target_i = torch.max(target_i, target_j.float())
        target.append(target_i)

    return target


def transform_preds(coords, center, scale, res, z_res=None, invert=1):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    for p in range(coords.size(0)):
        if coords.size(1) == 2:
            coords[p, 0:2] = to_torch(transform(coords[p, 0:2], center, scale, res, invert, 0))
        elif coords.size(1) == 3:
            coords[p, 0:3] = to_torch(transform3d(coords[p, 0:3], center, scale, res, z_res, invert, 0))
        else:
            # print('dimension not match.')
            raise Exception('dimension not match.')
    return coords


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

  
def readPtsTorch(ptspath):
    pts = []
    with open(ptspath, 'r') as file:
        lines = file.readlines()
        num_points = int(lines[1].split(' ')[1])
        for i in range(3,3+num_points):
            point = [float(num) for num in lines[i].split(' ')]
            pts.append(point)
    return torch.Tensor(pts)


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

# Bottleneck represents the Residual Module with layers (128X1X1) -> (128X3X3) -> (256X1X1) Fig.4 left
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


# A single hourglass
class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

# An hourglass network with multiple hourglasses with intermediate supervision
class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=[16]):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) 
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        if len(num_classes) == 1:
            num_classes = num_classes*num_stacks

        assert len(num_classes) == num_stacks

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes[i], kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes[i], ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) 
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    # forward pass
    def forward(self, x):
       
        out = []
        
        # Transform image from 256X256 to 64X64
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x = self.layer1(x)  
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x)  

        # number of stack = 4, 
        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_  # x = previous hourglass layer output +
                                      # fc layer + output of the current hourglass layer (shown in the diagram above Fig.4 Right)

        return out



class coordRegressor(nn.Module):
    def __init__(self, nParts):#, input_size, feat_size, output_size):
        super(coordRegressor, self).__init__()
        self.nParts = nParts
        self.depth = 5
        in_ch = [1,64,128,256,512]
        out_ch = [64,128,256,512,256]
        ec_k = [4,4,4,4,4]
        ec_s = [2,2,2,2,1]
        ec_p = [1,1,1,1,0]
        encoder_ = []

        # Adding the Conv layers to reduce the # of channels with every layer [64 -> 128 -> 256 -> 512 -> 256]
        for i in range(self.depth):
            if i < self.depth-1:
                layer_i = nn.Sequential(
                    nn.Conv3d(in_ch[i], out_ch[i], kernel_size=ec_k[i], stride=ec_s[i], padding=ec_p[i]),
                    nn.BatchNorm3d(out_ch[i]),
                    # nn.ReLU(inplace=True)
                    nn.LeakyReLU(0.2, inplace=True)
                    )
            else:
                layer_i = nn.Sequential(
                    nn.Conv3d(in_ch[i], out_ch[i], kernel_size=ec_k[i], stride=ec_s[i], padding=ec_p[i]),
                    # nn.ReLU(inplace=True)
                    # nn.Sigmoid()
                    )
            encoder_.append(layer_i)
        self.encoder = nn.ModuleList(encoder_)

        self.fc_cord = nn.Linear(out_ch[-1], self.nParts*3)
        
  # Start with a 64X64X64 volume (convolves to) -> 1X1X256 -> Fully Connected layer of (68 * 3 points) where 68 is number of landmark points and 3 is x,y,z dimensions.
    def forward(self, x):
        feat = x
        for i in range(self.depth):
            feat = self.encoder[i](feat)
            
        embedding = feat.view(x.size(0), -1)
        cord = self.fc_cord(embedding)
     
        
        return feat, embedding, cord

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def readPtsTorch(ptspath):
    pts = []
    with open(ptspath, 'r') as file:
        lines = file.readlines()
        num_points = int(lines[1].split(' ')[1])
        for i in range(3,3+num_points):
            point = [float(num) for num in lines[i].split(' ')]
            pts.append(point)

    return torch.tensor(pts)


def readPts(ptspath):
    pts = []
    with open(ptspath, 'r') as file:
        lines = file.readlines()
        num_points = int(lines[1].split(' ')[1])
        for i in range(3,3+num_points):
            point = [float(num) for num in lines[i].split(' ')]
            pts.append(point)

    return np.array(pts)
  
class JVCRDataSet(Dataset):
    """Custom Dataset for loading entries from HDF5 databases"""

    def __init__(self, path):
    
        self.path = path
        items = os.listdir(path)
        items.sort()

        datasetMap = {a:b for a,b in enumerate([i[:-4] for i in items if i.endswith('.jpg')])} # maps index with imageid
            
        self.datasetMap = datasetMap
      

    def __getitem__(self, index):
      return self.datasetMap[index]
     

    def __len__(self):
        return len(self.datasetMap) # return self.num_entries
  
def inputListTransform(imgList, ptsList):
    input_tensor_list = []
    for idx in range(0, len(imgList)):
        image = imgList[idx]
        pts = ptsList[idx]
        center, scale, bbox = boundingbox(to_numpy(pts))
        scale *= 1.25    
        l = center[0] - scale*200/2.
        u = center[1] - scale*200/2.
        w = scale*200
        h = scale*200    
        bbox = [l, u, l + w, u + h]
                
        lm_gt = torch.Tensor(pts)  
        image_tensor = torchvision.transforms.ToTensor()(image)  
        input_tensor_list.append(img_crop(image_tensor, center, scale))
        
    return torch.stack(input_tensor_list)
  
  
def reshapeTensorList(tensorList):
  outerList = []
  for idx1 in range(0,len(tensorList)):
    innerList = []
    for idx2 in range(0, len(tensorList[0])):
      innerList.append(tensorList[idx2][idx1].squeeze(0))
    outerList.append(torch.stack(innerList))
  return outerList

def save_model(model, optimizer, file_name):
    torch.save({
        'model' : model,
        'model_state_dict': model.state_dict(),
        'optimizer' : optimizer
    }, file_name)


num_stacks=1
num_blocks=4
num_classes=[1, 2, 4, 64]

  
train_type = "hourglass"
dataset_path = "imgs/"
train_dataset = JVCRDataSet(dataset_path)
batch_size = 4
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

modelRegression = coordRegressor(68)
modelHourGlass = HourglassNet(Bottleneck, num_blocks, num_stacks, num_classes)
learning_rate = 1e-4
loss_fn = torch.nn.MSELoss()#(reduction='sum')
optimizer = torch.optim.RMSprop(modelHourGlass.parameters(), lr = learning_rate)

num_epochs = 25
save_freq = 5
counter = 0
for epoch in range(num_epochs):

    for batch_idx, (x) in enumerate(train_loader):
        counter = counter + 1
        images=[]
        tensors=[]
        pts=[]

        for item in x:
          images.append(Image.open(dataset_path+item+".jpg").convert('RGB'))
          tensors.append(torch.load(dataset_path+item+".tensor"))
          pts.append(to_torch(readPtsTorch(dataset_path+item+".pts")))        
      
        images = inputListTransform(images, pts)
        tensors = reshapeTensorList(tensors)
        output = modelHourGlass(images)
        
        #compute losses
        loss_layer1 = loss_fn(output[0], tensors[0])  
        loss_layer2 = loss_fn(output[1], tensors[1])           
        loss_layer3 = loss_fn(output[2], tensors[2])   
        loss_layer4 = loss_fn(output[3], tensors[3])    

     
        total_loss = loss_layer1 + loss_layer2 + loss_layer3 + loss_layer4 # Adding the losses from the four layers in the Hourglass Network

        print("epoch",epoch, "batch_number", batch_idx, "loss", total_loss.item())
      
        modelHourGlass.zero_grad()

        total_loss.backward()
        optimizer.step()
              
    if epoch % save_freq == 0:
        file_name = 'epoch-'+str(epoch)+'-loss-+'+str(total_loss)+'+reg-model-opt.model'
        save_model(modelHourGlass, optimizer, file_name)
        print('model '+file_name+' saved..')


