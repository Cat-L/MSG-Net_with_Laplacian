import torch.nn as nn
import torch
import torch.nn.functional as F


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
        # print(self.weight)
        # print(input.size())
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)