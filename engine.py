from datatools import get_transforms
import torch
def train_single_epoch(model, dataset, device, criterion, optimizer, writer, epoch, train_loader):
    model.train()
                        
                        
    if dataset.input=='RGB':
        dataset.transforms=get_transforms(train=True,means=dataset.means[:3], stds=dataset.stds[:3])
    elif dataset.input=='D':
        dataset.transforms=get_transforms(train=True, means=dataset.means[3:], stds=dataset.stds[3:])
    elif dataset.input=='RGB-D': 
        dataset.transforms=get_transforms(train=True, means=dataset.means, stds=dataset.stds)
                        
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
    # training_val_loss=0

def validate(model, dataset, device, training_category, sav_dir, criterion, writer, epoch, val_loader, best_val_loss):
    current_val_loss=0
    # training_val_loss=0s           
                        
    model.eval()
    print('Validating and Checkpointing!')
                        
    if dataset.input=='RGB':
        dataset.transforms=get_transforms(train=False,means=dataset.means[:3], stds=dataset.stds[:3])
    elif dataset.input=='D':
        dataset.transforms=get_transforms(train=False, means=dataset.means[3:], stds=dataset.stds[3:])
    elif dataset.input=='RGB-D': 
        dataset.transforms=get_transforms(train=False,means=dataset.means, stds=dataset.stds)
                        
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
            
    if current_val_loss<best_val_loss or epoch==0:
        best_val_loss=current_val_loss
        torch.save(model.state_dict(), sav_dir+'bestmodel'+training_category+'_' + dataset.input + '_' + dataset.out + '.pth') # should be fixed! 
        print('Best model Saved! Val NMSE: ', str(best_val_loss))
        with open('run.txt', 'a') as f:
            f.write('\n')
            f.write('Best model Saved! Val NMSE: '+ str(best_val_loss))
        
    else:
        print('Model is not good (might be overfitting)! Current val NMSE: ', str(current_val_loss), 'Best Val NMSE: ', str(best_val_loss))
        with open('run.txt', 'a') as f:
                    f.write('\n')
                    f.write('Model is not good (might be overfitting)! Current val NMSE: '+ str(current_val_loss)+ 'Best Val NMSE: '+ str(best_val_loss))
    return best_val_loss