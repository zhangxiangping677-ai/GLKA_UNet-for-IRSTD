import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义深度可分卷积层
class DepthSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, padding=kernel_size // 2)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# 定义GLSA模块
class GLSA(nn.Module):
    def __init__(self, channels):
        super(GLSA, self).__init__()

        # 定义卷积层
        self.conv1x3 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.conv3x1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
        self.conv3x3 = nn.Conv2d(channels, channels, (3, 3), padding=1)

        # 定义深度可分卷积层
        self.dsc = DepthSeparableConv(channels, channels, 3)

    def forward(self, x):
        # 计算Avgp(X), Avgp(Y), Avgp(C) 和 Maxp(C)
        avg_x = F.adaptive_avg_pool2d(x, (x.size(2), 1))
        avg_y = F.adaptive_avg_pool2d(x, (1, x.size(3)))
        avg_c = F.adaptive_avg_pool2d(x, (1, 1))
        max_c = F.adaptive_max_pool2d(x, (1, 1))

        # 使用卷积层处理这些特征
        x1 = self.conv1x3(avg_x)
        x2 = self.conv3x1(avg_y)
        x3 = self.conv3x3(avg_c + max_c)

        # 广播机制操作
        x_final = x1 * x2 * x3

        # 深度可分卷积
        out_feat = self.dsc(x_final) * x

        return out_feat


# 创建一个GLSA实例并测试
if __name__ == "__main__":
    model = GLSA(4)
    input_tensor = torch.randn(1, 4, 32, 32)  # 假设输入大小为32x32
    output = model(input_tensor)
    print(output.shape)