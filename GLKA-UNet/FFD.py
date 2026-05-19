import torch
import torch.nn as nn


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


def Probility_refine(x1, x2):
    w = x1 * x2
    w_sum = x1 * x2 + (1. - x1) * (1. - x2)
    return w / (w_sum + 1e-6)


class FFD(nn.Module):
    def __init__(self, in_spatial_low, in_spatial_high, in_prior):
        super(FFD, self).__init__()
        self.conv_block_low = nn.Sequential(
            Vanila_Conv_no_pool(in_spatial_low, in_spatial_low // 16, 1),
            nn.Conv2d(in_spatial_low // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.conv_block_high = nn.Sequential(
            Vanila_Conv_no_pool(in_spatial_high, in_spatial_high // 16, 1),
            nn.Conv2d(in_spatial_high // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(3, 1, 1, padding=0),
            nn.Sigmoid())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_channel = nn.Sequential(
            nn.Conv2d(in_spatial_low, in_prior, 1, padding=0),
        )
        self.sig = nn.Sigmoid()

        self.Up_to_2 = nn.Upsample(scale_factor=2)

        self.conv_final = Vanila_Conv_no_pool(in_spatial_low, in_prior, 1)
        self.conv_concat = Vanila_Conv_no_pool(in_spatial_low + in_spatial_high, in_spatial_low, 1)

        self.final_attention = nn.Sequential(
            nn.Conv2d(in_prior, in_prior // 4, 3, 1, padding=1),
            nn.BatchNorm2d(in_prior // 4),
            nn.ReLU(),
            nn.Conv2d(in_prior // 4, 1, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x_spatial_low, x_spatial_high):
        b1, c1, w1, h1 = x_spatial_low.size()
        b2, c2, w2, h2 = x_spatial_high.size()
        if (w1, h2) != (w2, h2):
            x_spatial_high = self.Up_to_2(x_spatial_high)

        x_low_map = self.conv_block_low(x_spatial_low)
        x_high_map = self.conv_block_high(x_spatial_high)
        spatial_attention_map = self.conv_spatial(
            torch.cat([x_low_map, x_high_map, Probility_refine(x_low_map, x_high_map)], 1))

        x_spatial_low = self.conv_concat(torch.cat((x_spatial_low, x_spatial_high), 1)) * spatial_attention_map
        # x_spatial_low = self.Up_to_2(x_spatial_low)
        channel_attention = self.sig(
            self.conv_channel(self.avg_pool(x_spatial_low)) + self.conv_channel(self.max_pool(x_spatial_low)))

        out = self.conv_final(x_spatial_low) * channel_attention

        return out


if __name__ == '__main__':
    block = FFD(16, 32, 16)  # 输入通道数，输出通道数
    input1 = torch.rand(4, 16, 256, 256)  # 输入B C H W
    input2 = torch.rand(4, 32, 128, 128)
    output = block(input1, input2)
    print(output.size())