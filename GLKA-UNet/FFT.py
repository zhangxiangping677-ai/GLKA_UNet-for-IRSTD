import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class FFT(nn.Module):
    def __init__(self, feature_dim=512):
        super(FFT, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.df1 = nn.Sequential(
            nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.df1 = self.df1.to(self.device)

    def forward(self, x):
        B, C, H, W = x.size()
        # N = H * W
        # x_reshaped = x.view(B, C, N)
        low_freq_tensor, mid_freq_tensor, high_freq_tensor = self.filter_frequency_bands(x)
        # if H == 256 and W == 256:
        #     low_freq_tensor = low_freq_tensor.to("cpu")
        #     mid_freq_tensor = mid_freq_tensor.to("cpu")
        #     high_freq_tensor = high_freq_tensor.to("cpu")
        #     np.save('low_freq_tensor.npy', low_freq_tensor)
        #     np.save('mid_freq_tensor.npy', mid_freq_tensor)
        #     np.save('high_freq_tensor.npy', high_freq_tensor)
        #     high_freq_tensor = high_freq_tensor.to("cuda:0")
        #     mid_freq_tensor = mid_freq_tensor.to("cuda:0")
        #     low_freq_tensor = low_freq_tensor.to("cuda:0")
        out = low_freq_tensor + mid_freq_tensor + high_freq_tensor
        avg_attn = torch.mean(out, dim=1, keepdim=True)
        max_attn, _ = torch.max(out, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        agg = self.df1(agg)
        return agg

    def reshape_to_square(self, tensor):
        """
        Reshapes a tensor to a square shape.

        Args:
            tensor (torch.Tensor): The input tensor of shape (B, C, N), where B is the batch size,
                C is the number of channels, and N is the number of elements.

        Returns:
            tuple: A tuple containing:
                - square_tensor (torch.Tensor): The reshaped tensor of shape (B, C, side_length, side_length),
                  where side_length is the length of each side of the square tensor.
                - side_length (int): The length of each side of the square tensor.
                - side_length (int): The length of each side of the square tensor.
                - N (int): The original number of elements in the input tensor.
        """
        B, C, N = tensor.shape
        side_length = int(np.ceil(np.sqrt(N)))
        padded_length = side_length ** 2

        padded_tensor = torch.zeros((B, C, padded_length), device=self.device)
        padded_tensor[:, :, :N] = tensor

        square_tensor = padded_tensor.view(B, C, side_length, side_length)

        return square_tensor, side_length, side_length, N

    def filter_frequency_bands(self, tensor, cutoff=0.2):
        """
        Filters the input tensor into low, mid, and high frequency bands.

        Args:
            tensor (torch.Tensor): The input tensor to be filtered.
            cutoff (float, optional): The cutoff value for frequency band filtering.

        Returns:
            torch.Tensor: The low frequency band of the input tensor.
            torch.Tensor: The mid frequency band of the input tensor.
            torch.Tensor: The high frequency band of the input tensor.
        """

        tensor = tensor.float()
        # tensor, H, W, N = self.reshape_to_square(tensor)
        B, C, H, W = tensor.shape

        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)
        low_cutoff = max_radius * cutoff
        high_cutoff = max_radius * (1 - cutoff)

        fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1))

        def create_filter(shape, low_cutoff, high_cutoff, mode='band', device=self.device):
            rows, cols = shape
            center_row, center_col = rows // 2, cols // 2

            y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing='ij')
            distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)

            mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)

            if mode == 'low':
                mask[distance <= low_cutoff] = 1
            elif mode == 'high':
                mask[distance >= high_cutoff] = 1
            elif mode == 'band':
                mask[(distance > low_cutoff) & (distance < high_cutoff)] = 1

            return mask

        low_pass_filter = create_filter((H, W), low_cutoff, None, mode='low')[None, None, :, :]
        high_pass_filter = create_filter((H, W), None, high_cutoff, mode='high')[None, None, :, :]
        mid_pass_filter = create_filter((H, W), low_cutoff, high_cutoff, mode='band')[None, None, :, :]

        fft_tensor = fft_tensor.to(self.device)

        low_freq_fft = fft_tensor * low_pass_filter
        high_freq_fft = fft_tensor * high_pass_filter
        mid_freq_fft = fft_tensor * mid_pass_filter

        low_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        high_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        mid_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(mid_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real

        return low_freq_tensor, mid_freq_tensor, high_freq_tensor


class Freq_block(nn.Module):
    def __init__(self, dim, dfilter_freedom=[3, 2],
                 dfilter_type='piecewise_linear'):
        super().__init__()
        self.dim = dim
        self.dw_amp_conv = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.df1 = nn.Sequential(
            nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.df2 = nn.Sequential(
            nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.dw_pha_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, groups=dim*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
            )

    def forward(self, x):
        b, c, h, w = x.shape
        msF = torch.fft.rfft2(x+1e-8, dim=(-2, -1))
        msF = torch.cat([
            msF[:, :, msF.size(2) // 2 + 1:, :],
            msF[:, :, :msF.size(2) // 2 + 1, :]], dim=2)
        # msF = torch.fft.fftshift(msF, dim=(-2, -1))
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        amp_fuse = self.dw_amp_conv(msF_amp)
        avg_attn = torch.mean(amp_fuse, dim=1, keepdim=True)
        max_attn, _ = torch.max(amp_fuse, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        agg = self.df1(agg)
        amp_fuse = amp_fuse*agg
        amp_res = amp_fuse - msF_amp
        pha_guide = torch.cat((msF_pha,amp_res),dim=1)
        pha_fuse = self.dw_pha_conv(pha_guide)
        avg_attn = torch.mean(pha_fuse, dim=1, keepdim=True)
        max_attn, _ = torch.max(pha_fuse, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        agg = self.df2(agg)
        pha_fuse = pha_fuse * agg
        pha_fuse = pha_fuse*(2. * math.pi) - math.pi
        # pha_fuse = torch.clamp(pha_fuse, -math.pi, math.pi)
        ## amp_fuse = amp_fuse + msF_amp
        # pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        # out=torch.fft.ifftshift(out, dim=(-2, -1))
        out = torch.cat([
            out[:, :, out.size(2) // 2 - 1:, :],
            out[:, :, :out.size(2) // 2 - 1, :]], dim=2)
        out = torch.abs(torch.fft.irfft2(out+1e-8, s=(h, w)))
        if torch.isnan(out).sum() > 0:
            print('freq feature include NAN!!!!')
            # assert torch.isnan(out).sum() == 0  ##这里有问题
            out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        out = out + x
        return F.relu(out)


if __name__ == '__main__':
    model = FFT(3)
    x = torch.randn(1, 3, 224, 224)
    # y1, y2, y3 = model(x)
    # print(y1.shape)
    # print(y2.shape)
    # print(y3.shape)
    y = model(x)
    print(y.shape)