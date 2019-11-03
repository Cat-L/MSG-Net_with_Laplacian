from torch.utils.tensorboard import SummaryWriter

import os

import sys

import time
import numpy as np
import torchvision
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import net
import utils
from net import Net, Vgg16

from option import Options

np.random.seed(42)
torch.manual_seed(42)



transform = transforms.Compose([transforms.Scale(256),
																transforms.CenterCrop(256),
																transforms.ToTensor(),
																transforms.Lambda(lambda x: x.mul(255))])
train_dataset = datasets.ImageFolder("E:\COCO", transform)
train_loader = DataLoader(train_dataset, batch_size=4)

style_model = Net(ngf=128)
# Inspiration=net.Inspiration(C=None)
# Bottleneck=net.Bottleneck()
# UpBottlenech=net.UpBottleneck()


for batch_id, (x, _) in enumerate(train_loader):
	x = Variable(utils.preprocess_batch(x))
	break
x=torch.randn(x.size())

writer = SummaryWriter('./Result')
# torch.onnx.export(style_model,  x, "MSG_net.onnx", verbose=True)
# dummt_input=torch.randn(4,3,256)
model = torchvision.models.resnet18(False)
with writer:
	writer.add_graph(model, torch.rand([1,3,224,224]))

# with writer:
#     writer.add_graph(style_model, (x,))

