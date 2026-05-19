import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from model.L2SKNet.DCT import MultiSpectralAttentionLayer as DCT
from model.L2SKNet.FFT import FFT as FFT


class Conv_Attention(nn.Module):
    def __init__(self, channels):
        super(Conv_Attention, self).__init__()
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        x_init = self.dconv5_5(x)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        conv_att = self.conv(x)
        return conv_att

class Avg_ChannelAttention(nn.Module):
    def __init__(self, channels, r=4):
        super(Avg_ChannelAttention, self).__init__()
        self.avg_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # bz,C_out,h,w -> bz,C_out,1,1
            nn.Conv2d(channels, channels // r, 1, 1, 0, bias=False),  # bz,C_out,1,1 -> bz,C_out/r,1,1
            nn.BatchNorm2d(channels // r),
            nn.ReLU(True),
            nn.Conv2d(channels // r, channels, 1, 1, 0, bias=False),  # bz,C_out/r,1,1 -> bz,C_out,1,1
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.avg_channel(x)


class LLSKM(nn.Module):
    def __init__(self, channels, height, weight, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(LLSKM, self).__init__()
        # General CNN
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        # Channel Attention for $\theta$
        # if channels >= 64:
        #     self.attn = DCT(channels, height, weight)
        # else:
        #     self.attn = Avg_ChannelAttention(channels)
        # self.attn = DCT(channels, height, weight, freq_sel_method="top"+str(channels//4))
        self.attn = Avg_ChannelAttention(channels)
        self.kernel_size = kernel_size
        # self.conv_att = Conv_Attention(channels)
        # self.fft = FFT(channels)
        # self.conv_bn_relu = nn.Sequential(
        #     nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(True),
        # )
        # self.PConv = PConv(channels, channels, k=kernel_size, s=stride)
        self.fre = FFT(channels)

    def forward(self, x):
        # Feature result from a $k\times k$ General CNN
        out_normal = self.conv(x)
        # conv_att = self.conv_att(x)
        # low, mid, high = self.fft(x)
        # Channel Attention for $\theta_n$
        theta = self.attn(x)
        # f_out = self.fre(x)
        # f_out = f_out.to('cpu')

        # Sum up for each $k\times k$ CNN filter
        kernel_w1 = self.conv.weight.sum(2).sum(2)
        # Extend the $1\times 1$ to $k\times k$
        kernel_w2 = kernel_w1[:, :, None, None]
        # Filter the feature with $\textbf{W}_{sum}$
        out_center = F.conv2d(input=x, weight=kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)
        # Filter the feature with $\textbf{W}_{c}$
        center_w1 = self.conv.weight[:, :, self.kernel_size // 2, self.kernel_size // 2]
        center_w2 = center_w1[:, :, None, None]
        out_offset = F.conv2d(input=x, weight=center_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        # modify6
        # pconv = self.PConv(x)

        # The output feature of our Diff LSFM block
        out = (out_center - out_normal + theta * out_offset) # * f_out
        out = self.conv(out)
        return out


class LLSKM_d(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=1, bias=False):
        super(LLSKM_d, self).__init__()
        # General CNN
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        # Channel Attention for $\theta$
        self.attn = Avg_ChannelAttention(channels)
        self.kernel_size = kernel_size

    def forward(self, x):
        # Feature result from a $k\times k$ General CNN
        out_normal = self.conv(x)
        # Channel Attention for $\theta_n$
        theta = self.attn(x)

        # Sum up for each $k\times k$ CNN filter
        kernel_w1 = self.conv.weight.sum(2).sum(2)
        # Extend the $1\times 1$ to $k\times k$
        kernel_w2 = kernel_w1[:, :, None, None]
        # Filter the feature with $\textbf{W}_{sum}$
        out_center = F.conv2d(input=x, weight=kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)
        # Filter the feature with $\textbf{W}_{c}$
        center_w1 = self.conv.weight[:, :, self.kernel_size // 2, self.kernel_size // 2]
        center_w2 = center_w1[:, :, None, None]
        out_offset = F.conv2d(input=x, weight=center_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        # The output feature of our Diff LSFM block
        # $\textbf{Y} = {{\mathcal{W}}_s (\textbf{X})} = \mathcal{W}_{sum}(\textbf{X}) - {\mathcal{W}}(\textbf{X}) + \theta_c (\textbf{X})\circ {\mathcal{W}_{c}}{(\textbf{X})}$
        return out_center - out_normal + theta * out_offset


if __name__ == '__main__':
    net = LLSKM(16, 256, 256)
    x = torch.randn(4, 16, 256, 256)
    y = net(x)
    print(y.shape)