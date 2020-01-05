from torchfile import load as load_lua

import os,time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from net import Vgg16
from tqdm import tqdm

DATASET="L:\COCO"
VGG_MODEL_DIR="L:/vgg_model"
STYLE_FOLDER="L:/style"
MODEL_SAVE_DIR="L:/model_save"

def init_vgg16(model_folder):
    """load the vgg16 model feature"""
    if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
            os.system(
                'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7  ' + os.path.join(
                    model_folder, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'), force_8bytes_long=True)
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))


class LapConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(LapConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                        padding_mode)
        kernel = [[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]]

        self.weight = nn.Parameter(
            torch.Tensor(kernel).expand(out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=False)
        # self.kernel = torch.Tensor(kernel).expand(out_channels, in_channels // groups, 3)
        # self.weight=nn.Parameter(self.kernel,requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def my_forward(model, x):
    mo = nn.Sequential(*list(model.children())[:-1])
    feature = mo(x)
    feature = feature.view(x.size(0), -1)
    output = model.fc(feature)
    return feature, output

#
# DATASET = "L:\COCO"
# VGG_MODEL_DIR = "L:/vgg_model"
# STYLE_FOLDER = "F:\images\9styles"
# CONTENT_FOLDER = "F:\images\content"
# MODEL_SAVE_DIR = "L:/model_save"
# IMAGE_DIR="F:\images\\"

if __name__ == '__main__':
    # dataset="F:\COCO"
    # # 载入数据集
    # transform = transforms.Compose([transforms.Scale(256),
    #                                 transforms.CenterCrop(256),
    #                                 transforms.ToTensor(),
    #                                 transforms.Lambda(lambda x: x.mul(255))])
    # train_dataset = datasets.ImageFolder(dataset, transform)
    # train_loader = DataLoader(train_dataset, batch_size=2)

    # file1 = 'F:\images\content\\flowers.jpg'
    # file2 = 'F:\images\9styles\candy.jpg'
    #
    # # print(os.path.splitext(os.path.split(file1)[-1])[0])
    # filepath = 'F:\model_save\Final_epoch_2_Wed_Nov_13_06_16_15_2019_1.0_100.0.model'
    # model_name= os.path.splitext(os.path.split(filepath)[-1])[0]
    # print(os.path.join(IMAGE_DIR,model_name))
    # os.mkdir(path=os.path.join(IMAGE_DIR,model_name))
    for i in  tqdm(range(10000000000),dynamic_ncols=True):
        print(i)