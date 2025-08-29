import torch.nn as nn
import torch

class stu_net3(nn.Module):
    def __init__(self, input_nc=2, output_nc=1,filters1=None,bias1=None,filters2=None,bias2=None):
        super(stu_net3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=0),
            );
        print(filters1)
        if filters1 is not None and bias1 is not None:
            self.conv1[1].weight = nn.Parameter(filters1,requires_grad=True)
            self.conv1[1].bias = nn.Parameter(bias1,requires_grad=True)
        self.act1 = nn.Sequential(nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=0),
            );
        if filters2 is not None and bias2 is not None:
            self.conv2[1].weight = nn.Parameter(filters2,requires_grad=True)
            self.conv2[1].bias = nn.Parameter(bias2,requires_grad=True)
        self.act2= nn.Sequential(nn.Tanh())
    def encoder(self, input):
        G11 = self.conv1(input)
        G11_1 = self.act1(G11)
        return [G11_1,G11]
    def decoder(self, f_en):
        G21 = self.conv2(f_en[0]);
        G21_1 = self.act2(G21)
        print(G21_1.shape)
        return [G21_1,G21]
    def forward(self, input):
        return self.decoder(self.encoder(input)[0])[0]
