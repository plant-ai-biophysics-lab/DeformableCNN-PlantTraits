#%%
from torch.functional import split
from datatools import *
from engine import train_single_epoch, validate
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
    
    
    
    
dataset = GreenhouseDataset(rgb_dir=RGB_Data_Dir, d_dir=Depth_Data_Dir, jsonfile_dir=JSON_Files_Dir, transforms=transform_type) 
dataset.df= dataset.df.iloc[:-50]



train_split, val_split = train_test_split(dataset.df, test_size=0.2, random_state=split_seed, stratify=dataset.df['Variety'])

train = torch.utils.data.Subset(dataset, train_split.index.tolist())
val   = torch.utils.data.Subset(dataset, val_split.index.tolist())
dataset.set_indices(train.indices, val.indices)
                                                                                     
                                
train_loader = torch.utils.data.DataLoader(train, batch_size=15, num_workers=12, shuffle=True)#, sampler=train_sampler)
val_loader   = torch.utils.data.DataLoader(val,   batch_size=15, shuffle=False, num_workers=12)#, sampler=val_sampler)


#=============================================================================================================#
#======================================= Optimization and loss function  =====================================#
#=============================================================================================================#                                
params = [p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.Adam(params, lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)                         
criterion=NMSELoss()


#=============================================================================================================#
#===============================================- Training ===================================================#
#=============================================================================================================#                                
for In in inputs:
    for Out in outputs:
        dataset.input=In
        dataset.out  =Out                        

        device=torch.device('cuda')
        model= GreenhouseMidFusionRegressor(input_data_type=In, num_outputs=NumOutputs, conv_type=ConvType)
        model.to(device)
    
                                
        best_val_loss=9999999 # initial dummy value
        current_val_loss=0
        # training_val_loss=0
           
        writer = SummaryWriter()
        start=time.time()
                                
        for epoch in range(num_epochs):
            with open('run.txt', 'a') as f:
                f.write('\n')
                f.write('Epoch: '+ str(epoch+1)+ ', Time Elapsed: '+ str((time.time()-start)/60)+' mins')
            print('Epoch: ', str(epoch+1), ', Time Elapsed: ', str((time.time()-start)/60),' mins')

            train_single_epoch(model, dataset, device, criterion, optimizer, writer, epoch, train_loader)

            best_val_loss=validate(model, dataset, device, training_category, sav_dir, criterion, writer, epoch, val_loader, best_val_loss)
        # scheduler.step()
        
# %%

                                
                                
