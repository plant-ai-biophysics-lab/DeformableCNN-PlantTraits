import os
import numpy as np
import torch

import torchvision
import albumentations as A
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

import random
from random import Random
from datetime import datetime
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from scipy import stats

'''
Simple dataset starter to take in an image (r,g,b,n) in a given set and return

'''
#%%
def flatten_json(nested_json, exclude=['']):
    """Flatten json object with nested keys into a single level.
        Args:
            nested_json: A nested json object.
            exclude: Keys to exclude from output.
        Returns:
            The flattened json object if successful, None otherwise.
    """
    out = {}

    def flatten(x, name='', exclude=exclude):
        if type(x) is dict:
            for a in x:
                if a not in exclude: flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)

    return out
    
def trainval_split(dataset, val_fraction=0.7, split_seed=0):

    validation_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - validation_size

    train, val = torch.utils.data.random_split(
        dataset, 
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(split_seed)
        )

    return train, val

#%%


class GreenhouseDataset(Dataset):
    def __init__(self, rgb_dir, d_dir, jsonfile_dir, transforms=None, input='RGB-D',output='All'):

        self.df= pd.read_json(jsonfile_dir)
        # flatten_json is a costum function to flat the nested json files!
        
        self.transforms = transforms
        self.rgb_dir=rgb_dir
        self.d_dir=d_dir
        self.input=input
        self.output=output
        

    def __getitem__(self, idx):
        # load images 
        row=self.df.iloc[idx]

        if self.input=='RGB':
            rgbd = plt.imread(self.rgb_dir+row['image'])
        if self.input=='D':
            rgbd = plt.imread(self.d_dir+row['depth_image'])
            rgbd=np.expand_dims(rgbd,2)
        if self.input=='RGB-D': 
            rgb = plt.imread(self.rgb_dir+row['image'])
            depth = plt.imread(self.d_dir+row['depth_image'])
            rgbd = np.dstack([rgb,depth])
        
        # load GT regression data
        if self.out=='ALL':
            target=list(row['outputs']['regression'].values())
        else:
            target=[row['outputs']['regression'][self.out]]



        #make sure your img and mask array are in this format before passing into albumentations transforms, img.shape=[H, W, C]
        if self.transforms is not None:
            aug = self.transforms(image=rgbd)
            rgbd = aug['image']

        rgbd = np.transpose(rgbd, (2,0,1))

        #pytorch wants a different format for the image ([C, H, W])
        rgbd = torch.as_tensor(rgbd, dtype=torch.float32)
        target=torch.as_tensor(target, dtype=torch.float32)

        return rgbd, target

    def __len__(self):
        return len(self.df)



## FIGURE OUT HOW TO CROP ALL THE IMAGES TO GET RID OF EXTRANIOUS PIXELS
def get_transforms(train, means, stds):
    if train:
        transforms = A.Compose([
        # A.Crop(x_min=650, y_min=200, x_max=1450, y_max=900, always_apply=False, p=1.0),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-5, 5), interpolation=0, border_mode=0, value=means, mask_value=None),
        A.Normalize(mean=means, std=stds, max_pixel_value=1.0, always_apply=False, p=1.0)
        ])
    else:
        transforms =  A.Compose([
        # A.Crop(x_min=650, y_min=200, x_max=1450, y_max=900, always_apply=False, p=1.0),
        A.Normalize(mean=means, std=stds, max_pixel_value=1.0, always_apply=False, p=1.0)
        ])
    return transforms



def get_RGB_transforms(train):
    if train:
        transforms = A.Compose([
        # A.Crop(x_min=650, y_min=200, x_max=1450, y_max=900, always_apply=False, p=1.0),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-5, 5), interpolation=0, border_mode=0, value=(0.5482, 0.4620, 0.3602), mask_value=None),
        A.Normalize(mean=(0.5482, 0.4620, 0.3602), std=(0.1639, 0.1761, 0.2659), max_pixel_value=1.0, always_apply=False, p=1.0)
   
        ])
    else:
        transforms =  A.Compose([
        # A.Crop(x_min=650, y_min=200, x_max=1450, y_max=900, always_apply=False, p=1.0),
        A.Normalize(mean=(0.5482, 0.4620, 0.3602), std=(0.1639, 0.1761, 0.2659), max_pixel_value=1.0, always_apply=False, p=1.0)    
        ])
    return transforms



def get_D_transforms(train):
    if train:
        transforms = A.Compose([
        # A.Crop(x_min=650, y_min=200, x_max=1450, y_max=900, always_apply=False, p=1.0),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-5, 5), interpolation=0, border_mode=0, value=(0.0142), mask_value=None),
        A.Normalize(mean=(0.0142), std=(0.0036), max_pixel_value=1.0, always_apply=False, p=1.0)
        ])
    else:
        transforms =  A.Compose([
        # A.Crop(x_min=650, y_min=200, x_max=1450, y_max=900, always_apply=False, p=1.0),
        A.Normalize(mean=(0.0142), std=(0.0036), max_pixel_value=1.0, always_apply=False, p=1.0)
        
        ])
    return transforms
# %%
