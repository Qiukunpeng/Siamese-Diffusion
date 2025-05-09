import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)


class ResidualBlock(nn.Module):
    def __init__(self, dims, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_nd(dims, in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.silu = nn.SiLU()
        self.conv2 = conv_nd(dims, out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.residual = conv_nd(dims, in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.silu(out)
        out = self.conv2(out)
        out += residual
        out = self.silu(out)
        return out


class PatchMerging(nn.Module):
    def __init__(self, patch_dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.patch_dim = patch_dim
        self.norm = norm_layer(4 * patch_dim)
        self.reduction = nn.Linear(4 * patch_dim, 2 * patch_dim)

    def forward(self, x):
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Height and Width must be even."

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(B, H // 2, W // 2, 2 * C)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, hint_channels, dim=2):
        super(FeatureExtractor, self).__init__()
        self.initial_conv = conv_nd(dim, hint_channels, 16, kernel_size=3, stride=1, padding=1)
        self.layer1 = ResidualBlock(dim, 16, 16)
        self.conv_before_res2 = conv_nd(dim, 16, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = ResidualBlock(dim, 32, 32)
        self.patch_merge1 = PatchMerging(patch_dim=32)
        self.conv_before_res3 = conv_nd(dim, 64, 64, kernel_size=3, stride=1, padding=1)
        self.layer3 = ResidualBlock(dim, 64, 64)
        self.patch_merge2 = PatchMerging(patch_dim=64)
        self.conv_before_res4 = conv_nd(dim, 128, 128, kernel_size=3, stride=1, padding=1)
        self.layer4 = ResidualBlock(dim, 128, 128)
        self.patch_merge3 = PatchMerging(patch_dim=128)
        self.conv_before_res5 = conv_nd(dim, 256, 256, kernel_size=3, stride=1, padding=1)
        self.layer5 = ResidualBlock(dim, 256, 256)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.conv_before_res2(x)
        x = self.layer2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.patch_merge1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_before_res3(x)
        x = self.layer3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.patch_merge2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_before_res4(x)
        x = self.layer4(x)
        x = x.permute(0, 2, 3, 1)
        x = self.patch_merge3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_before_res5(x)
        x = self.layer5(x)
        return x
