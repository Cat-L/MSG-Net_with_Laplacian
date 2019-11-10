import torchvision
import  os
from net import Vgg16
from torchfile import load as load_lua


import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms

from net import Net, Vgg16




def init_vgg16(model_folder):
    """load the vgg16 model feature"""
    if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
            os.system(
                'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7  ' + os.path.join(model_folder, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'), force_8bytes_long=True)
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))

class LapConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(LapConv2d,self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        kernel = [[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]]


        self.weight=nn.Parameter(torch.Tensor(kernel).expand(out_channels,in_channels//groups,kernel_size,kernel_size),requires_grad=False)
        # self.kernel = torch.Tensor(kernel).expand(out_channels, in_channels // groups, 3)
        # self.weight=nn.Parameter(self.kernel,requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def my_forward(model, x):
    mo = nn.Sequential(*list(model.children())[:-1])
    feature = mo(x)
    feature = feature.view(x.size(0), -1)
    output= model.fc(feature)
    return feature, output

if __name__ == '__main__':

    dataset="F:\COCO"
    # 载入数据集
    transform = transforms.Compose([transforms.Scale(256),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=2)


    # # vgglua = load_lua(os.path.join("F:\\vgg_model", 'vgg16.t7'), force_8bytes_long=True)
    # # print(type(vgglua))
    # # vgg = Vgg16()
    # # for (src, dst) in zip(vgglua, vgg.parameters()):
    # #     dst.data[:] = src
    #
    for batch_id, (x, _) in enumerate(train_loader):
        a=x
        break


    vgg_=torchvision.models.vgg16(pretrained=False)
    vgg_.load_state_dict(torch.load("vgg16-00b39a1b.pth"))
    vgg_=vgg_.features

    mo1 = nn.Sequential(*list(vgg_.children())[:4])
    mo2 = nn.Sequential(*list(vgg_.children())[:9])
    mo3 = nn.Sequential(*list(vgg_.children())[:19])
    mo4 = nn.Sequential(*list(vgg_.children())[:23])

    features=[]

    features.append(mo1(x).view(x.size(0), -1))
    features.append(mo2(x).view(x.size(0), -1))
    features.append(mo3(x).view(x.size(0), -1))
    features.append(mo4(x).view(x.size(0), -1))


    print(features)