#%%
from scipy.stats.stats import power_divergence
# from datatools import GreenhouseDataset, get_transforms, trainval_split
from datatools import GreenhouseDataset, GreenhouseSIMOMidFusionDataset, get_RGB_transforms, trainval_split, get_D_transforms

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

test_loader = torch.utils.data.DataLoader(testset, batch_size=50,num_workers=0, shuffle=False)


device=torch.device('cuda')

cri=NMSELoss()
mse=nn.MSELoss()
inputs=['RGB','D']

with torch.no_grad():
    # for image, target in test_loader:
    #     image=image.to(device)
    #     target=target.to(device)

    #     preds=model(image)
    #     nmse, pred=cri(preds, target) 
    for inp in inputs:
        testset.input=inp
        model= GreenhouseMidFusionRegressor(input=inp, num_outputs=5, conv_type='deformable')
        model.to(device)
        model.load_state_dict(torch.load('./experiments/DCN_midfusionresnet18_SIMO_'+inp+'.pth'))
        model.eval()
        if inp=='RGB':
            testset.transforms=get_RGB_transforms(train=False)
        if inp=='D':
            testset.transforms=get_D_transforms(train=False)
        for im, targets in test_loader:
            im=im.to(device)
            targets=targets.to(device)
            preds=model(im)
            # nmse=criterion(preds, targets)
            nmse, pred=cri(preds, targets)
        print(mse(preds[:,0],targets[:,0]).tolist())
        print(mse(preds[:,1],targets[:,1]).tolist())
        print(mse(preds[:,2],targets[:,2]).tolist())
        print(mse(preds[:,3],targets[:,3]).tolist())
        print(mse(preds[:,4],targets[:,4]).tolist())
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
