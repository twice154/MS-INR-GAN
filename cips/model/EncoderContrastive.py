__all__ = ['ContrastiveEncoder',
           'ContrastivePredictor', 
          ]

import math

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ConvLayer, ResBlock, EqualLinear


class ContrastiveEncoder(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], input_size=3, n_first_layers=0, **kwargs):
        super().__init__()

        self.input_size = input_size

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(input_size, channels[size], 1)]
        convs.extend([ConvLayer(channels[size], channels[size], 3) for _ in range(n_first_layers)])

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        # self.stddev_group = 4
        # self.stddev_feat = 1

        # self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        # self.final_linear = nn.Sequential(
        #     EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
        #     EqualLinear(channels[4], 1),
        # )

    def forward(self, input, key=None):
        multi_scale_encoder_feat = []

        out = input
        for i in range(len(self.convs)):
            out = self.convs[i](out)
            multi_scale_encoder_feat.append(out)
        
        return multi_scale_encoder_feat
        # out = self.convs(input)

        # batch, channel, height, width = out.shape
        # group = min(batch, self.stddev_group)
        # stddev = out.view(
        #     group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        # )
        # stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        # stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        # stddev = stddev.repeat(group, 1, height, width)
        # out = torch.cat([out, stddev], 1)

        # out = self.final_conv(out)

        # out = out.view(batch, -1)
        # out = self.final_linear(out)

        # return out


class ContrastivePredictor(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], input_size=3, n_first_layers=0, **kwargs):
        super().__init__()

        self.input_size = input_size

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # convs = [ConvLayer(input_size, channels[size], 1)]
        # convs.extend([ConvLayer(channels[size], channels[size], 3) for _ in range(n_first_layers)])
        convs = []

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            # out_channel = channels[2 ** (i - 1)]

            convs.append(nn.Sequential(ConvLayer(in_channel, in_channel, 1), ConvLayer(in_channel, in_channel, 1)))
            # convs.append(ConvLayer(in_channel, in_channel, 1))

            in_channel = channels[2 ** (i - 1)]

        self.convs = nn.Sequential(*convs)

        # self.stddev_group = 4
        # self.stddev_feat = 1

        # self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        # self.final_linear = nn.Sequential(
        #     EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
        #     EqualLinear(channels[4], 1),
        # )

    def forward(self, input, key=None):
        multi_scale_predictor_feat = []

        for i in range(len(self.convs)):
            multi_scale_predictor_feat.append(self.convs[i](input[i]))
        
        return multi_scale_predictor_feat
        # out = self.convs(input)

        # batch, channel, height, width = out.shape
        # group = min(batch, self.stddev_group)
        # stddev = out.view(
        #     group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        # )
        # stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        # stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        # stddev = stddev.repeat(group, 1, height, width)
        # out = torch.cat([out, stddev], 1)

        # out = self.final_conv(out)

        # out = out.view(batch, -1)
        # out = self.final_linear(out)

        # return out