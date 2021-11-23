import torch
import torch.nn as nn

# nmse loss
class NMSELoss(nn.Module):
    def __init__(self):
          # super(diceloss, self).init()
          super(NMSELoss, self).__init__()
          # print('HI')
    def forward(self, pred, target):
        if target.size() != pred.size():
              raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), pred.size()))
          
        num=torch.sum((target-pred)**2,0)
        den=torch.sum(target**2,0)




        return torch.sum(num/den)

