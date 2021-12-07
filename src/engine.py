from datatools import get_RGB_transforms, get_D_transforms, get_transforms
import torch
def train(model, In, dataset, device, criterion, optimizer, writer, epoch, train_loader):
    model.train()
                        
                        
    if In=='RGB':
        dataset.transforms=get_RGB_transforms(train=True)
    elif In=='D':
        dataset.transforms=get_D_transforms(train=True)
    elif In=='RGB-D': 
        dataset.transforms=get_transforms(train=True)
                        
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

def validate(model, In, dataset, device, criterion, writer, epoch, val_loader, best_val_loss):
    current_val_loss=0
    # training_val_loss=0s           
                        
    model.eval()
    print('val')
                        
    if In=='RGB':
        dataset.transforms=get_RGB_transforms(train=False)
    elif In=='D':
        dataset.transforms=get_D_transforms(train=False)
    elif In=='RGB-D': 
        dataset.transforms=get_transforms(train=False)
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
        torch.save(model.state_dict(), './experiments1/DCN_midfusionresnet18_SISO_' + In + '_' + Out + '.pth') # should be fixed! 
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