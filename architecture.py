#%%
import torch
import torchvision
import torch.nn as nn
import torchvision.ops as ops
import copy


class DeformConv2d(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        n_offsets=1):

        super(DeformConv2d, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=groups
        self.bias=bias
        self.n_offsets=n_offsets

        self.deformable_layer= ops.DeformConv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
        self.offset_layer=nn.Conv2d(self.in_channels, 2*self.n_offsets*self.kernel_size[0]*self.kernel_size[1], self.kernel_size, self.stride, self.padding,self.dilation,self.groups, self.bias)

    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        # self.offset_in=x
        self.x=x

        offset=self.offset_layer(x)
        self.offsets=offset.detach().cpu()
        h = offset.register_hook(self.activations_hook)
        o=self.deformable_layer(x,offset)
        # h = o.register_hook(self.activations_hook)
        return o
    
    def weight_replace(self, weights):
        self.deformable_layer.weight=weights

    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.offset_layer(x)




class GreenhouseMidFusionRegressor(torch.nn.Module):
    def __init__(self, input='RGB-D',num_outputs=5, conv_type='standard'):
        super(GreenhouseMidFusionRegressor, self).__init__()
        self.input=input

        if self.input=='RGB-D':
            self.rgbencoder= torchvision.models.resnet18(pretrained=True)
            self.depthencoder= torchvision.models.resnet18(pretrained=True)
            if conv_type=='deformable':
                weights_tmp = self.rgbencoder.conv1.weight
                self.rgbencoder.conv1=DeformConv2d(3,64,(7,7), (2,2), (3,3), bias=False, n_offsets=3)
                self.rgbencoder.conv1.weight_replace(weights_tmp)

                self.rgbencoder.layer1=self.deform(self.rgbencoder.layer1)
                self.rgbencoder.layer2=self.deform(self.rgbencoder.layer2)
                self.rgbencoder.layer3=self.deform(self.rgbencoder.layer3)
                self.rgbencoder.layer4=self.deform(self.rgbencoder.layer4)

                self.depthencoder.conv1=DeformConv2d(3,64,(7,7), (2,2), (3,3), bias=False, n_offsets=3)
                self.depthencoder.conv1.weight_replace(weights_tmp)
                self.depthencoder.layer1=self.deform(self.depthencoder.layer1)
                self.depthencoder.layer2=self.deform(self.depthencoder.layer2)
                self.depthencoder.layer3=self.deform(self.depthencoder.layer3)
                self.depthencoder.layer4=self.deform(self.depthencoder.layer4)
                self.depthencoder=nn.Sequential(DeformConv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, n_offsets=1), nn.ReLU(),self.depthencoder)
            else:
                self.depthencoder=nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(), self.depthencoder)


            self.first_linear=nn.Linear(2000, 1000)
            self.second_linear=nn.Linear(1000, 500)
            self.final_linear=nn.Linear(500, num_outputs)
        if self.input=='RGB' or self.input=='D':
            self.encoder= torchvision.models.resnet18(pretrained=True)
            weights_tmp = self.encoder.conv1.weight
            if conv_type=='deformable': 
                self.encoder.conv1=DeformConv2d(3,64,(7,7), (2,2), (3,3), bias=False, n_offsets=3)
                self.encoder.conv1.weight_replace(weights_tmp)
                self.encoder.layer1=self.deform(self.encoder.layer1)
                self.encoder.layer2=self.deform(self.encoder.layer2)
                self.encoder.layer3=self.deform(self.encoder.layer3)
                self.encoder.layer4=self.deform(self.encoder.layer4)

            if self.input=='D':
                if conv_type=='deformable':
                    self.encoder=nn.Sequential(DeformConv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),nn.ReLU, self.encoder)
                else:
                    self.encoder=nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),nn.ReLU, self.encoder)
            self.first_linear=nn.Linear(1000, 512)
            self.second_linear=nn.Linear(512, 256)
            self.final_linear=nn.Linear(256, num_outputs)

        self.dropout=nn.Dropout(p=0.05)
        self.relu= nn.ReLU()
        self.prelu=nn.PReLU()


    def forward(self, inp):
        if self.input=='RGB-D':
            x_rgb=inp[:,:3,:,:]
            x_depth=inp[:,3:,:,:]
            x_rgb=self.rgbencoder(x_rgb)
            x_depth=self.depthencoder(x_depth)
            x=torch.cat((x_rgb, x_depth), dim=1)
        else:
            x=self.encoder(inp)

        x=self.dropout(x)
        x=self.relu(x)
        x=self.first_linear(x)
        x=self.dropout(x)
        x=self.relu(x)
        x=self.second_linear(x)
        x=self.dropout(x)
        x=self.relu(x)
        x=self.final_linear(x)
        x=self.prelu(x)
        return x

    def deform(self, layer):
        new=copy.deepcopy(layer)
        for i, b in enumerate(layer):
            for c, operation in zip(b._modules,b.children()):
                if isinstance(operation,nn.Conv2d):
                    # print(c)
                    weights_tmp = operation.weight
                    l=DeformConv2d(operation.in_channels, operation.out_channels, operation.kernel_size, operation.stride, operation.padding, bias=operation.bias, n_offsets=8)
                    l.weight_replace(weights_tmp)
                    setattr(new[i], c, l)

        return new


