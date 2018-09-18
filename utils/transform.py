#Utils SET 2

import os
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
    