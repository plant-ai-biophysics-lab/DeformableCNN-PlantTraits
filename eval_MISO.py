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


test_loader = torch.utils.data.DataLoader(testset, batch_size=50,num_workers=0, shuffle=False)


device=torch.device('cuda')


cri=NMSELoss()
mse=nn.MSELoss()

outputs=['FW','DW','H','D','LA']

final=torch.zeros((50,0))

all_targets=torch.zeros((50,0))

with torch.no_grad():
    # for image, target in test_loader:
    #     image=image.to(device)
    #     target=target.to(device)

    #     preds=model(image)
    #     nmse, pred=cri(preds, target)


    for o in outputs:
        testset.out=o
        model= GreenhouseMidFusionRegressor(input='RGB-D',num_outputs=1, conv_type='deformable')
        model.to(device)
        model.load_state_dict(torch.load('./experiments/DCN_midfusionresnet18_'+o+'.pth'))
        model.eval()
        for rgbd, targets in test_loader:
                rgbd=rgbd.to(device)
                targets=targets.to(device)
                preds=model(rgbd)
                mse_loss=mse(preds, targets)
                # nmse=criterion(preds, targets)
                nmse, pred=cri(preds, targets)
                final=torch.cat((final, preds.detach().cpu()),1)
                all_targets=torch.cat((all_targets, targets.detach().cpu()),1)
    
                
        # break
        print('MISO ',o,' MSE loss: ', mse_loss.tolist())

    f_nmse, pred=cri(final, all_targets)

    


# %%
