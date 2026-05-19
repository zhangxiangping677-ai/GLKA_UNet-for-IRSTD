import torch
import torch.nn as nn
import torch.nn.functional as F
from model.L2SKNet.KAN import KAN as KAN


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        # 定义卷积层1：输入通道in_channels -> 隐藏通道hidden_dim
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        # 定义卷积层2：隐含通道hidden_dim -> 输出通道out_channels
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)

        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
        self.KAN = KAN(hidden_dim)

    def forward(self, x):
        # 通过第一层卷积 + ReLU激活
        x = F.relu(self.conv1(x))
        # 使用Dropout进行正则化
        # x = self.dropout(x)
        x = self.KAN(x)
        # 通过第二层卷积
        x = self.conv2(x)
        return x