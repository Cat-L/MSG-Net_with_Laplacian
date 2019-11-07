import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from net import LapNet



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

class test_layer(nn.Module):

    def __init__(self):
        super(test_layer,self).__init__()
        self.model1=nn.Sequential(
         nn.AvgPool2d(2),
         LapConv2d(3, 3),
        )
        self.model2=nn.Sequential(
            nn.AvgPool2d(4),
            LapConv2d(3, 3)
        )
        self.model3=nn.Sequential(
            nn.AvgPool2d(8),
            LapConv2d(3,3)
        )



    def forward(self, input):

        y=self.model1(input)
        y=self.model2(y)
        y=self.model3(y)
        return y


if __name__ == '__main__':
    # kernel = [[0, -1, 0],
    #           [-1, 4, -1],
    #           [0, -1, 0]]
    #
    #
    # lr=0.001
    # # optimizer = Adam(style_model.parameters(), lr)
    # mse_loss = torch.nn.MSELoss()
    #
    # style_loader = utils.StyleLoader("D:\Work\Project\Python\\NST\PyTorch-Multi-Style-Transfer-master\experiments\images\9styles", 512)
    # style_=style_loader.get(2)
    #
    # test1=test_layer()
    # # print(test1(style_))
    #
    #
    #
    # dataset="F:\COCO"
    # # 载入数据集
    # transform = transforms.Compose([transforms.Scale(256),
    #                                 transforms.CenterCrop(256),
    #                                 transforms.ToTensor(),
    #                                 transforms.Lambda(lambda x: x.mul(255))])
    # train_dataset = datasets.ImageFolder(dataset, transform)
    # train_loader = DataLoader(train_dataset, batch_size=2)
    #
    #
    # result=[]
    # for batch_id, (x, _) in enumerate(train_loader):
    #     # print(x.size)
    #     temp=test1(x)
    #     result.append(temp)
    #     # print(temp)
    #     if batch_id==2:
    #         break
    #
    # b=1
    # for i in result[1]:
    #    b*=i
    # a=(2*mse_loss(result[0],result[1]))/torch.numel(result[1])
    # print(a.data)
    CONTENT_LAYERS = ['relu4_2', 'relu5_2']
    content_features = {}
for layer in CONTENT_LAYERS:
    content_features[layer] =1

print(str(5e2))