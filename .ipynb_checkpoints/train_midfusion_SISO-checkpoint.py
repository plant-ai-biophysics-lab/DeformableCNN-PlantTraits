#%%

from torch.functional import split
from datatools import GreenhouseDataset, get_D_transforms, get_RGB_transforms
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

dataset=GreenhouseDataset(rgb_dir='/data/pvraja/greenhouse-data/RGBImages/',d_dir='/data/pvraja/greenhouse-data/DepthImages/',jsonfile_dir='/data/pvraja/greenhouse-data/GroundTruth/GroundTruth_All_388_Images.json', transforms=get_RGB_transforms(train=False))
dataset.df=dataset.df.iloc[:-50]


split_seed=12

train_split, val_split = train_test_split(dataset.df, test_size=0.2, random_state=split_seed, stratify=dataset.df['Variety'])

train = torch.utils.data.Subset(dataset, train_split.index.tolist())
val = torch.utils.data.Subset(dataset, val_split.index.tolist())

dataset.set_indices(train.indices, val.indices)


train_loader = torch.utils.data.DataLoader(train, batch_size=15,num_workers=12, shuffle=True)#, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val, batch_size=15, shuffle=False, num_workers=12)#, sampler=val_sampler)



#%%

# this part is just to check the MEAN and STD of the dataset (dont run unless you need mu and sigma)



# nimages = 0
# mean = 0.
# std = 0.
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=12)

# for batch, _ in dataloader:
#     # Rearrange batch to be the shape of [B, C, W * H]
#     batch = batch.view(batch.size(0), batch.size(1), -1)
#     # Update total number of images
#     nimages += batch.size(0)
#     # Compute mean and std here
#     mean += batch.mean(2).sum(0) 
#     std += batch.std(2).sum(0)

# # Final step
# mean /= nimages
# std /= nimages

# print('Mean: '+ str(mean))
# print('Standard Deviation', str(std))




# %%


device=torch.device('cuda')
inputs=['RGB','D']
outputs=['FW','DW','H','D','LA']



for inp in inputs:
    for out in outputs:
        dataset.input=inp
        dataset.out=out
        
        model= GreenhouseMidFusionRegressor(input=inp, num_outputs=1, conv_type='deformable')
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]

        optimizer=torch.optim.Adam(params, lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        criterion=NMSELoss()


        num_epochs=400

        best_val_loss=None # initial dummy value

        current_val_loss=0
        # training_val_loss=0
        model.eval()
        if inp=='RGB':
            dataset.transforms=get_RGB_transforms(train=False)
        if inp=='D':
            dataset.transforms=get_D_transforms(train=False)
        with torch.no_grad():
            for i, (rgbd, targets) in enumerate(val_loader):
                # print(rgb.shape, d.shape)

                rgbd=rgbd.to(device)
                # d=d.to(device)
                targets=targets.to(device)
                preds=model(rgbd)

                

                loss=criterion(preds, targets)
                # acc=nmse(preds.detach(), targets)

                current_val_loss=current_val_loss+loss.tolist()
                # training_val_loss=training_val_loss+loss.detach().cpu().numpy()
                

            best_val_loss=current_val_loss
        


        writer = SummaryWriter()
        start=time.time()
        for epoch in range(num_epochs):
            with open('run.txt', 'a') as f:
                f.write('\n')
                f.write('Epoch: '+ str(epoch+1)+ ', Time Elapsed: '+ str((time.time()-start)/60)+' mins')
            print('Epoch: ', str(epoch+1), ', Time Elapsed: ', str((time.time()-start)/60),' mins')
            model.train()
            
            if inp=='RGB':
                dataset.transforms=get_RGB_transforms(train=True)
            if inp=='D':
                dataset.transforms=get_D_transforms(train=True)
            for i, (rgbd, targets)  in enumerate(train_loader):
                rgbd=rgbd.to(device)
                # d=d.to(device)
                targets=targets.to(device)
                preds=model(rgbd)

                loss=criterion(preds, targets)
                # with torch.no_grad():
                #     acc=nmse(preds.detach(), targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Train NMSE: ', str(loss.tolist()))
                with open('run.txt', 'a') as f:
                    f.write('\n')
                    f.write('Train NMSE: '+ str(loss.tolist()))
                # writer.add_scalar("MSE Loss/train", loss, (epoch*train_loader.sampler.num_samples+i)/train_loader.sampler.num_samples)
                writer.add_scalar("NMSE Loss/train", loss, (epoch*train_loader.sampler.num_samples+i)/train_loader.sampler.num_samples)
            current_val_loss=0
            # training_val_loss=0
            model.eval()
            print('val')
            if inp=='RGB':
                dataset.transforms=get_RGB_transforms(train=False)
            if inp=='D':
                dataset.transforms=get_D_transforms(train=False)
            with torch.no_grad():
                for i, (rgbd, targets) in enumerate(val_loader):

                    rgbd=rgbd.to(device)
                    # d=d.to(device)
                    targets=targets.to(device)
                    preds=model(rgbd)
                    

                    loss=criterion(preds, targets)
                    # acc=nmse(preds.detach(), targets)

                    current_val_loss=current_val_loss+loss.tolist()
                    # training_val_loss=training_val_loss+loss.detach().cpu().numpy()

                # writer.add_scalar("MSE Loss/val", training_val_loss, epoch)
                writer.add_scalar("NMSE Loss/val", current_val_loss, epoch)
                    
            if current_val_loss<best_val_loss or best_val_loss==None:
                best_val_loss=current_val_loss
                torch.save(model.state_dict(), './experiments1/DCN_midfusionresnet18_SISO_'+inp+'_'+out+'.pth')
                print('Best model Saved! Val NMSE: ', str(best_val_loss))
                with open('run.txt', 'a') as f:
                    f.write('\n')
                    f.write('Best model Saved! Val NMSE: '+ str(best_val_loss))
                
            else:
                print('Model is not good (might be overfitting)! Current val NMSE: ', str(current_val_loss), 'Best Val NMSE: ', str(best_val_loss))
                with open('run.txt', 'a') as f:
                    f.write('\n')
                    f.write('Model is not good (might be overfitting)! Current val NMSE: '+ str(current_val_loss)+ 'Best Val NMSE: '+ str(best_val_loss))
        # scheduler.step()
        
# %%
