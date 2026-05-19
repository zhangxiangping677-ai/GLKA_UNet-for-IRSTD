import torch.nn as nn
from model.L2SKNet.LLSKMs import LLSKM
import torch
from model.L2SKNet.FFN import FeedForwardNetwork
from model.L2SKNet.DCT import MultiSpectralAttentionLayer as DCT
from model.L2SKNet.KAN import KAN as KAN
from model.L2SKNet.MLP import ChannelMLP as MLP


class O_ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(O_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class O_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(O_SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, self.bias, padding=(self.act_num * 2 + 1) // 2, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=(self.act_num * 2 + 1) // 2, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class Vanila_Conv_no_pool(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation(c2, act_num=3)
        # self.act = self.default_act if ahaoct is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class MPE(nn.Module):
    def __init__(self, in_channel, height, weight):
        super(MPE, self).__init__()
        self.Conv_1 = nn.Sequential(
            LLSKM(in_channel // 4, height, weight, 1, 1, padding=0),
            nn.BatchNorm2d(in_channel // 4),
            nn.LeakyReLU()
        )

        self.Conv_3 = nn.Sequential(
            LLSKM(in_channel // 4, height, weight, 3, 1, padding=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.LeakyReLU()
        )

        self.Conv_5 = nn.Sequential(
            LLSKM(in_channel // 4, height, weight, 5, 1, padding=2),
            nn.BatchNorm2d(in_channel // 4),
            nn.LeakyReLU()
        )

        self.Conv_7 = nn.Sequential(
            LLSKM(in_channel // 4, height, weight, 7, 1, padding=3),
            nn.BatchNorm2d(in_channel // 4),
            nn.LeakyReLU()
        )

        self.Conv = Vanila_Conv_no_pool(in_channel, in_channel, 1)
        # self.fusion = LinAngularXCA_CA(in_channel)
        # self.ca = ChannelAttention(in_channel)
        self.dct = DCT(in_channel, height, weight)
        self.sa = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.se = SE_Block(in_channel)
        # self.dyt = DynamicTanh([in_channel, height, weight])
        self.FFN = FeedForwardNetwork(in_channel, in_channel, in_channel*2)
        self.KAN = KAN(in_channel)
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)

        # Batch Normalization
        self.bn = nn.BatchNorm2d(in_channel)

        # ReLU activation
        self.relu = nn.ReLU()
        self.mlp = MLP(in_channel)
        self.ca = O_ChannelAttention(in_channel)
        self.sa = O_SpatialAttention()

    def forward(self, x):
        b, c, h, w = x.size()
        x_1 = x[:, :(c // 4), :, :]
        x_2 = x[:, (c // 4):(c // 4) * 2, :, :]
        x_3 = x[:, (c // 4) * 2:(c // 4) * 3, :, :]
        x_4 = x[:, (c // 4) * 3:, :, :]

        x_4_7 = self.Conv_7(x_4)
        x_3_5 = self.Conv_5(x_3)
        x_2_3 = self.Conv_3(x_2)
        x_1_1 = self.Conv_1(x_1)

        # out = self.se(self.Conv(torch.cat((x_1_1, x_2_3, x_3_5, x_4_7, x), 1)))
        # out = self.fusion(torch.cat((x_1_1, x_2_3, x_3_5, x_4_7), 1), x)
        # out = self.Conv(torch.cat((x_1_1, x_2_3, x_3_5, x_4_7, x), 1))
        out = torch.cat((x_1_1, x_2_3, x_3_5, x_4_7), 1)
        # out = x * self.sigmoid(out) + x
        # out = self.sa(out) * out
        # ---------------- best  -----------------------
        # out1 = self.KAN(out)
        # out = self.relu(self.bn(self.conv(out1))) + out
        # out = self.relu(self.conv(self.se(out)))
        # ---------------- best  -----------------------
        # out1 = self.FFN(out)
        # out = out1 + out

        out1 = self.relu(self.dwconv(out))
        out1 = self.KAN(out1)
        out1 = self.ca(out1) * out1
        out1 = self.sa(out1) * out1
        out1 = self.relu(self.dwconv(out1))

        out = out1 + out
        out = self.relu(self.conv(out))

        return out


if __name__ == '__main__':
    model = MPE(16, 16, 16)
    x = torch.randn(4, 16, 16, 16)
    y = model(x)
    print(y.shape)