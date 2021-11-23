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


# dataset=GreenhouseDataset(image_dir='/data2/greenhouse-data/train-images/',jsonfile_dir='/data2/greenhouse-data/train-gt/GroundTruth.json', transforms=get_transforms(train=False))
# dataset=GreenhouseMISOMidFusionDataset(image_dir='/data2/greenhouse-data/train-images/',jsonfile_dir='/data2/greenhouse-data/train-gt/GroundTruth.json', transforms=get_transforms(train=False))

# split_seed=12

# train, val = trainval_split(dataset, val_fraction=0.25, split_seed=split_seed)

# dataset.set_indices(train.indices, val.indices)


# testset=GreenhouseDataset(image_dir='/data2/greenhouse-data/train-images/',jsonfile_dir='/data2/greenhouse-data/train-gt/GroundTruth_SendJuly6.json', transforms=get_transforms(train=False))
testset=GreenhouseDataset(rgb_dir='/data/pvraja/greenhouse-data/RGBImages/',d_dir='/data/pvraja/greenhouse-data/DepthImages/',jsonfile_dir='/data/pvraja/greenhouse-data/GroundTruth/GroundTruth_All_388_Images.json', transforms=get_RGB_transforms(train=False))

# testset.means=dataset.means
# testset.stds=dataset.stds
testset.df=testset.df[-50:]
# tes

test_loader = torch.utils.data.DataLoader(testset, batch_size=50,num_workers=0, shuffle=False)


device=torch.device('cuda')
# model=GreenhouseRegressor()

cri=NMSELoss()
mse=nn.MSELoss()
# diff_df = pd.merge(dataset.df, testset.df, how='outer', indicator=True, on='RGBImage')
# diff_df = diff_df[diff_df._merge != 'both']

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

# pred=preds

#%%
Measurements = {}

for i in range(pred.shape[0]):
    print(testset.df.iloc[i]['RGBImage'])
    # print(pred[i][0]) #Freshweight
    # print(pred[i][1]) #Dry weight
    # print(pred[i][2]) # height
    image_name = 'Image'+str(i+1)
    dic = {}
    dic['RGBImage'] = testset.df.iloc[i]['RGBImage']
    dic['DebthInformation'] = testset.df.iloc[i]['DebthInformation']
    dic['FreshWeightShoot'] = pred[i][0].tolist()
    dic['DryWeightShoot'] = pred[i][1].tolist()
    dic['Height'] = pred[i][2].tolist()
    dic['Diameter'] = pred[i][3].tolist()
    dic['LeafArea'] = pred[i][4].tolist()
    Measurements[image_name] = dic
master={}
master['Measurements']=Measurements

with open('predict.json', 'w') as fp:
    json.dump(master, fp)

    


# %%
