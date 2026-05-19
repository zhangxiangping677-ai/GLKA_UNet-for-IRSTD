import torch
import torch.nn as nn


class ChannelMLP(nn.Module):
    def __init__(self, channels, hidden_dim=None):
        super(ChannelMLP, self).__init__()
        hidden_dim = channels * 2
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # [b, h, w, c]
        x = x.reshape(b * h * w, c)  # [b*h*w, c]
        x = self.mlp(x)  # 对通道进行 MLP
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [b, c, h, w]
        return x