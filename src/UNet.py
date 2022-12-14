"""PyTorch implementation of a U-Net.

Adapted by Matt Guay from https://discuss.pytorch.org/t/unet-implementation/426

"""
import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=2,
            depth=5,
            n_init_filters=32,
            padding=True,
            instance_norm=True,
            up_mode='upconv',
            leaky=True):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            n_init_filters (int): number of filters in the first layer
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            instance_norm (bool): Use instance normalization after layers with
                            an activation function.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
            leaky (bool): If True, use LeakyReLU activation instead of ReLU.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth

        # Keep track of each layer's input (previous) channel size
        prev_channels = in_channels

        # U-Net encoder module
        self.encoder_module = nn.ModuleList()
        for i in range(depth):
            self.encoder_module.append(
                UNetConvBlock(
                    prev_channels,
                    n_init_filters * 2**i,
                    padding,
                    instance_norm,
                    leaky))
            prev_channels = n_init_filters * 2**i

        # U-Net decoder module, Skip connections are handled in the
        # `UNetDecoderBlock`
        self.decoder_module = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.decoder_module.append(
                UNetDecoderBlock(
                    prev_channels,
                    n_init_filters * 2**i,
                    up_mode,
                    padding,
                    instance_norm,
                    leaky))
            prev_channels = n_init_filters * 2**i

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        encoder_blocks = []
        for i, down in enumerate(self.encoder_module):
            x = down(x)
            if i != len(self.encoder_module) - 1:
                encoder_blocks.append(x)
                x = F.max_pool2d(x, 2)

        # Pass in x and the output of the corresponding encoder conv block
        for i, up in enumerate(self.decoder_module):
            x = up(x, encoder_blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(
            self,
            in_size,
            out_size,
            padding,
            instance_norm,
            leaky):
        super(UNetConvBlock, self).__init__()
        block = []

        conv = nn.Conv2d
        activation = nn.LeakyReLU if leaky else nn.ReLU

        block.append(conv(
            in_size,
            out_size,
            kernel_size=3,
            padding=int(padding)))
        block.append(activation())

        block.append(conv(
            out_size,
            out_size,
            kernel_size=3,
            padding=int(padding)))
        block.append(activation())
        if instance_norm:
            block.append(nn.InstanceNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetDecoderBlock(nn.Module):
    def __init__(
            self,
            in_size,
            out_size,
            up_mode,
            padding,
            instance_norm,
            leaky):
        super(UNetDecoderBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(
            in_size,
            out_size,
            padding,
            instance_norm,
            leaky)

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


def center_crop(layer, target_size):
    _, _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[
            :,
            :,
            diff_y:(diff_y + target_size[0]),
            diff_x:(diff_x + target_size[1])]
