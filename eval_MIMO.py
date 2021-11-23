#%%
from scipy.stats.stats import power_divergence
# from datatools import GreenhouseDataset, get_transforms, trainval_split
from datatools import GreenhouseDataset, get_transforms, trainval_split

import torch, torchvision
# from architecture import GreenhouseRegressor
from architecture import GreenhouseMidFusionRegressor
from nmse import NMSELoss

from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import json




testset=GreenhouseDataset(rgb_dir='/data/pvraja/greenhouse-data/RGBImages/',d_dir='/data/pvraja/greenhouse-data/DepthImages/',jsonfile_dir='/data/pvraja/greenhouse-data/GroundTruth/GroundTruth_All_388_Images.json', transforms=get_transforms(train=False))

testset.df=testset.df[-50:]

test_loader = torch.utils.data.DataLoader(testset, batch_size=1,num_workers=0, shuffle=False)


device=torch.device('cuda')
model= GreenhouseMidFusionRegressor(input='RGB-D',num_outputs=5, conv_type='deformable')
model.to(device)
model.load_state_dict(torch.load('./DCN/midfusionresnet18_offsets8_MIMO_RELU1-3/epoch110.pth'))

model.eval()
cri=NMSELoss()
mse=nn.MSELoss()

with torch.no_grad():
    # for image, target in test_loader:
    #     image=image.to(device)
    #     target=target.to(device)

    #     preds=model(image)
    #     nmse, pred=cri(preds, target)
    # ap=torch.zeros((0,5),device=device)
    # at=torch.zeros((0,5),device=device)
    for i,(rgbd, targets) in enumerate(test_loader):
        rgbd=rgbd.to(device)
        targets=targets.to(device)
        preds=model(rgbd)
        nmse=cri(preds, targets)
        # nmse, pred=cri(preds, targets)
        # ap=torch.cat((ap, preds), 0)
        # at=torch.cat((at, targets), 0)
    # nmse, pred=cri(ap, at)
    
    print(mse(preds[:,0],targets[:,0]).tolist())
    print(mse(preds[:,1],targets[:,1]).tolist())
    print(mse(preds[:,2],targets[:,2]).tolist())
    print(mse(preds[:,3],targets[:,3]).tolist())
    print(mse(preds[:,4],targets[:,4]).tolist())
# pred=preds




# %%
