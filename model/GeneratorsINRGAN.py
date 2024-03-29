__all__ = ['INRGAN'
           ]

import math

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConstantInput, LFF, StyledConv, ToRGB, PixelNorm, EqualLinear, StyledResBlock


class INRGAN(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, **kwargs):
        super(INRGAN, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = LFF(hidden_size)
        # self.emb = ConstantInput(hidden_size, size=size)

        self.channels = {
            # 0: 512,
            # 1: 512,
            # 2: 512,
            # 3: 512,
            # 4: 256 * channel_multiplier,
            # 5: 128 * channel_multiplier,
            # 6: 64 * channel_multiplier,
            # 7: 32 * channel_multiplier,
            # 8: 16 * channel_multiplier,
            0: 32,
            1: 32,
            2: 32,
            3: 32,
            4: 16 * channel_multiplier,
            5: 8 * channel_multiplier,
            6: 3 * channel_multiplier,
            7: 2 * channel_multiplier,
            8: 1 * channel_multiplier,
        }

        multiplier = 1
        in_channels = int(self.channels[0])
        self.conv1 = StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = 4 # int(math.log(size, 2))

        self.n_intermediate = self.log_size - 1
        self.to_rgb_stride = 2
        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]
            self.linears.append(StyledConv(in_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            self.linears.append(StyledConv(out_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            self.to_rgbs.append(ToRGB(out_channels, style_dim, upsample=False))

            in_channels = out_channels

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

    def forward(self,
                coords,
                coords2,
                coords3,
                latent,
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                ):

        latent = latent[0]

        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent:
            latent = self.style(latent)

        x = self.lff(coords)

        # batch_size, _, w, h = coords.shape
        # if self.training:
        #     emb = self.emb(x)
        # else:
        #     emb = F.grid_sample(
        #         self.emb.input.expand(batch_size, -1, -1, -1),
        #         coords.permute(0, 2, 3, 1).contiguous(),
        #         padding_mode='border', mode='bilinear',
        #     )

        # x = torch.cat([x, emb], 1)

        rgb = 0

        x = self.conv1(x, latent)
        for i in range(self.n_intermediate):

            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
            x = torch.cat((x, coords2), 1)
            
            for j in range(self.to_rgb_stride):
                x = self.linears[i*self.to_rgb_stride + j](x, latent)

            rgb = self.to_rgbs[i](x, latent, rgb)

        if return_latents:
            return rgb, latent
        else:
            return rgb, None