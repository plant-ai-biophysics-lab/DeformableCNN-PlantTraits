#%%
from scipy.stats.stats import power_divergence
# from datatools import GreenhouseDataset, get_transforms, trainval_split
from datatools import GreenhouseDataset

import torch, torchvision
# from architecture import GreenhouseRegressor
from architecture import GreenhouseMidFusionRegressor
from nmse import NMSELoss

from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import json



testset=GreenhouseDataset(rgb_dir='/data/pvraja/greenhouse-data/RGBImages/',d_dir='/data/pvraja/greenhouse-data/DepthImages/',jsonfile_dir='/data/pvraja/greenhouse-data/GroundTruth/GroundTruth_All_388_Images.json', transforms=get_RGB_transforms(train=False))


testset.df=testset.df[-50:]
# tes

test_loader = torch.utils.data.DataLoader(testset, batch_size=50,num_workers=0, shuffle=False)


device=torch.device('cuda')
# model=GreenhouseRegressor()

cri=NMSELoss()
mse=nn.MSELoss()


inp='D'
outputs=['FW','DW','H','D','LA']

final=torch.zeros((50,0))

all_targets=torch.zeros((50,0))

with torch.no_grad():
    # for image, target in test_loader:
    #     image=image.to(device)
    #     target=target.to(device)

    #     preds=model(image)
    #     nmse, pred=cri(preds, target)

    if inp=='RGB':
        testset.transforms=get_RGB_transforms(train=False)
    if inp=='D':
        testset.transforms=get_D_transforms(train=False)
    for o in outputs:
        testset.input=inp
        testset.out=o
        model= GreenhouseMidFusionRegressor(input=inp, num_outputs=1, conv_type='deformable')

        model.to(device)
        model.load_state_dict(torch.load('./experiments1/DCN_midfusionresnet18_SISO_'+inp+'_'+o+'.pth'))
        model.eval()
        # ap=torch.zeros((0,1))
        # at=torch.zeros((0,1))
        for rgbd, targets in test_loader:
            rgbd=rgbd.to(device)
            targets=targets.to(device)
            preds=model(rgbd)
            # mse_loss=mse(preds, targets)
            # nmse=criterion(preds, targets)
            # nmse, pred=cri(preds, targets)
            # ap=torch.cat((ap, preds.detach().cpu()), 0)
            # at=torch.cat((at, targets.detach().cpu()), 0)
        final=torch.cat((final, preds.detach().cpu()),1)
        all_targets=torch.cat((all_targets, targets.detach().cpu()),1)
        mse_loss=mse(preds,targets)
                
        # break
        print('SISO ',o,' MSE loss: ', mse_loss.tolist())

    f_nmse=cri(final, all_targets)
