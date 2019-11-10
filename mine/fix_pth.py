import torch
import torchvision
import torch.nn as nn
import os
from net import Vgg16

def fix():
    sd = torch.load("F:\\vgg_model\\vgg16-00b39a1b.pth")
    sd['classifier.0.weight'] = sd['classifier.1.weight']
    sd['classifier.0.bias'] = sd['classifier.1.bias']
    del sd['classifier.1.weight']
    del sd['classifier.1.bias']

    sd['classifier.3.weight'] = sd['classifier.4.weight']
    sd['classifier.3.bias'] = sd['classifier.4.bias']
    del sd['classifier.4.weight']
    del sd['classifier.4.bias']

    torch.save(sd, "vgg16-00b39a1b.pth")

if __name__ == '__main__':
    wetdic=torch.load("vgg16-00b39a1b.pth")
    #
    # for i in wetdic:
    #     print(i)
    vgg16=torchvision.models.vgg16(pretrained=False)
    vgg16=vgg16.features

    print(vgg16,"\n\n\n")

    mo1 = nn.Sequential(*list(vgg16.children())[:4])
    mo2 = nn.Sequential(*list(vgg16.children())[4:9])
    mo3 = nn.Sequential(*list(vgg16.children())[9:16])
    mo4 = nn.Sequential(*list(vgg16.children())[16:23])

    print(mo1,"\n\n",mo2,"\n\n",mo3,"\n\n",mo4,"\n\n\n")