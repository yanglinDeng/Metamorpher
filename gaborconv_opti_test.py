import torch
import torch.nn as nn
import torch.nn.functional as F
import math
NUMBER = 0
# 检查CUDA是否可用
device = torch.device("cuda:"+str(NUMBER) if torch.cuda.is_available() else "cpu")

class gabor_net(nn.Module):
    def __init__(self, input_nc=2, output_nc=1, middle_nc=4):
        super(gabor_net, self).__init__()
        self.sigma = nn.Parameter(torch.rand(12, device=device), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(12, device=device), requires_grad=True)
        self.lamda = nn.Parameter(torch.rand(12, device=device), requires_grad=True)
        self.gamma = nn.Parameter(torch.rand(12, device=device), requires_grad=True)
        self.psi = nn.Parameter(torch.randn(12, device=device), requires_grad=True)
        self.bias_1 = nn.Parameter(torch.randn(4, device=device), requires_grad=True)
        self.bias_2 = nn.Parameter(torch.randn(1, device=device), requires_grad=True)




    def forward(self, input,conv1_weight,conv2_weight):
        input = input.to(device)
        G11 = F.conv2d(input, conv1_weight, padding=1, stride=1,bias=self.bias_1)
        G11_1 = torch.relu(G11)
        G21 = F.conv2d(G11_1, conv2_weight, padding=1, stride=1,bias=self.bias_2)
        G21_1 = torch.tanh(G21)
        return [G21_1, G21, G11_1, G11]



