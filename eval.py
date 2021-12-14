#%%
from torch.functional import split
from datatools import *
from engine import train, validate
import torch, torchvision
from architecture import GreenhouseMidFusionRegressor
from nmse import NMSELoss
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.optim import lr_scheduler
import os



testset_size=50   
sav_dir='./DCN/'
if not os.path.exists(sav_dir):
    os.mkdir(sav_dir)
RGB_Data_Dir   = '/data/pvraja/greenhouse-data/RGBImages/'
Depth_Data_Dir = '/data/pvraja/greenhouse-data/DepthImages/'  
JSON_Files_Dir = '/data/pvraja/greenhouse-data/GroundTruth/GroundTruth_All_388_Images.json'
split_seed=12    


num_epochs    =  400
ConvType      = 'deformable'
training_category = 'MIMO' #'MIMO', 'MISO', 'SIMO', 'SISO'


if training_category   == 'MIMO':
    transform_type = get_transforms(train=False) 
    inputs=['RGB-D']
    outputs=['All']
    NumOutputs    = 5
    
elif training_category == 'MISO':
    transform_type = get_transforms(train=False)
    inputs=['RGB-D']
    outputs=['FW','DW','H','D','LA']
    NumOutputs    = 1
    
elif training_category == 'SIMO':
    transform_type = get_RGB_transforms(train=False)
    inputs=['RGB','D']
    outputs=['All']
    NumOutputs    = 5
    
elif training_category == 'SISO':
    transform_type = get_RGB_transforms(train=False)
    
    inputs =['RGB','D']
    outputs=['FW','DW','H','D','LA']
    NumOutputs    = 1
    
    
    



testset = GreenhouseDataset(rgb_dir=RGB_Data_Dir, d_dir=Depth_Data_Dir, jsonfile_dir=JSON_Files_Dir, transforms=transform_type) 



testset.df=testset.df[-50:]
# tes

test_loader = torch.utils.data.DataLoader(testset, batch_size=50,num_workers=0, shuffle=False)


device=torch.device('cuda')
# model=GreenhouseRegressor()

cri=NMSELoss()
mse=nn.MSELoss()




with torch.no_grad():
    for In in inputs:
        final=torch.zeros((testset_size,0))
        all_targets=torch.zeros((testset_size,0))
        for Out in outputs:
            print('Input is ', In)
            testset.input=In
            testset.out  =Out                        

            device=torch.device('cuda')
            model= GreenhouseMidFusionRegressor(input_data_type=In, num_outputs=NumOutputs, conv_type=ConvType)
            model.to(device)
            model.load_state_dict(torch.load(sav_dir+'bestmodel'+training_category+'_' + In + '_' + Out + '.pth'))
            model.eval()


            if Out=='All':
                ap=torch.zeros((0,5),device=device)
                at=torch.zeros((0,5),device=device)
            else:
                ap=torch.zeros((0,1),device=device)
                at=torch.zeros((0,1),device=device)

            for rgbd, targets in test_loader:
                rgbd=rgbd.to(device)
                targets=targets.to(device)
                preds=model(rgbd)
                # mse_loss=mse(preds, targets)
                # nmse=criterion(preds, targets)
                # nmse, pred=cri(preds, targets)
                ap=torch.cat((ap, preds.detach().cpu()), 0)
                at=torch.cat((at, targets.detach().cpu()), 0)
            if Out=='All':
                print('FW MSE: ', str(mse(ap[:,0],at[:,0]).tolist()))
                print('DW MSE: ', str(mse(ap[:,1],at[:,1]).tolist()))
                print('H MSE: ', str(mse(ap[:,2],at[:,2]).tolist()))
                print('D MSE: ', str(mse(ap[:,3],at[:,3]).tolist()))
                print('LA MSE: ', str(mse(ap[:,4],at[:,4]).tolist()))
                print('Overall NMSE: ', str(cri(ap,at).tolist()))
            else:
                final=torch.cat((final, ap.detach().cpu()),1)
                all_targets=torch.cat((all_targets, at.detach().cpu()),1)
                print(Out,' MSE: ', str(mse(ap,at).tolist()))
        if Out=='All':
            print('Overall NMSE: ', str(cri(ap,at).tolist()))
        else:
            print('Overall NMSE: ', str(cri(final,all_targets).tolist()))
        
            
                                    
         