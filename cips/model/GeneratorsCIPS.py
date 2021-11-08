__all__ = ['CIPSskip',
           'CIPSres',
           'CIPSskippatch',
           'CIPSskippatchearly',
           'CIPSskippatchscale',
           'CIPSskippatchmip',
           'CIPSskippatchpgpe'
           ]

import math

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConstantInput, LFF, StyledConv, ToRGB, PixelNorm, EqualLinear, StyledResBlock, LFFMip


class CIPSskip(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, **kwargs):
        super(CIPSskip, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = LFF(hidden_size)
        self.emb = ConstantInput(hidden_size, size=size)

        self.channels = {
            0: 512,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
        }

        multiplier = 2
        in_channels = int(self.channels[0])
        self.conv1 = StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

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

        batch_size, _, w, h = coords.shape
        if self.training and w == h == self.size:
            emb = self.emb(x)
        else:
            emb = F.grid_sample(
                self.emb.input.expand(batch_size, -1, -1, -1),
                coords.permute(0, 2, 3, 1).contiguous(),
                padding_mode='border', mode='bilinear',
            )

        x = torch.cat([x, emb], 1)

        rgb = 0

        x = self.conv1(x, latent)
        for i in range(self.n_intermediate):
            for j in range(self.to_rgb_stride):
                x = self.linears[i*self.to_rgb_stride + j](x, latent)

            rgb = self.to_rgbs[i](x, latent, rgb)

        if return_latents:
            return rgb, latent
        else:
            return rgb, None


class CIPSres(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, **kwargs):
        super(CIPSres, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = LFF(int(hidden_size))
        self.emb = ConstantInput(hidden_size, size=size)

        self.channels = {
            0: 512,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 64 * channel_multiplier,
            8: 32 * channel_multiplier,
        }

        self.linears = nn.ModuleList()
        in_channels = int(self.channels[0])
        multiplier = 2
        self.linears.append(StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                       activation=activation))

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]
            self.linears.append(StyledResBlock(in_channels, out_channels, 1, style_dim, demodulate=demodulate,
                                               activation=activation))
            in_channels = out_channels

        self.to_rgb_last = ToRGB(in_channels, style_dim, upsample=False)

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

        batch_size, _, w, h = coords.shape
        if self.training and w == h == self.size:
            emb = self.emb(x)
        else:
            emb = F.grid_sample(
                self.emb.input.expand(batch_size, -1, -1, -1),
                coords.permute(0, 2, 3, 1).contiguous(),
                padding_mode='border', mode='bilinear',
            )
        out = torch.cat([x, emb], 1)

        for con in self.linears:
            out = con(out, latent)

        out = self.to_rgb_last(out, latent)

        if return_latents:
            return out, latent
        else:
            return out, None


class CIPSskippatch(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, **kwargs):
        super(CIPSskippatch, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = LFF(hidden_size)
        # self.emb = ConstantInput(hidden_size, size=size)

        self.channels = {
            0: 512,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
            # 0: 32,
            # 1: 32,
            # 2: 32,
            # 3: 32,
            # 4: 16 * channel_multiplier,
            # 5: 8 * channel_multiplier,
            # 6: 3 * channel_multiplier,
            # 7: 2 * channel_multiplier,
            # 8: 1 * channel_multiplier,
        }

        multiplier = 1
        in_channels = int(self.channels[0])
        self.conv1 = StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

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
            for j in range(self.to_rgb_stride):
                x = self.linears[i*self.to_rgb_stride + j](x, latent)

            rgb = self.to_rgbs[i](x, latent, rgb)

        if return_latents:
            return rgb, latent
        else:
            return rgb, None


class CIPSskippatchearly(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, **kwargs):
        super(CIPSskippatchearly, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = LFF(hidden_size)
        # self.emb = ConstantInput(hidden_size, size=size)

        self.channels = {
            0: 512,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
            # 0: 32,
            # 1: 32,
            # 2: 32,
            # 3: 32,
            # 4: 16 * channel_multiplier,
            # 5: 8 * channel_multiplier,
            # 6: 3 * channel_multiplier,
            # 7: 2 * channel_multiplier,
            # 8: 1 * channel_multiplier,
        }

        multiplier = 1
        in_channels = int(self.channels[0])
        self.conv1 = StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.to_to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

        self.n_intermediate = self.log_size - 1
        self.to_rgb_stride = 2
        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]
            self.linears.append(StyledConv(in_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            self.linears.append(StyledConv(out_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            self.to_rgbs.append(ToRGB(out_channels, style_dim, upsample=False))

            self.to_to_rgbs.append(ToRGB(out_channels, style_dim, upsample=False))

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
                latent,
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                early_exit=False,
                second_exit=False
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
            for j in range(self.to_rgb_stride):
                x = self.linears[i*self.to_rgb_stride + j](x, latent)

            if i == 0 and early_exit:
                rgb = self.to_to_rgbs[i](x, latent, rgb)

                if return_latents:
                    return rgb, latent
                else:
                    return rgb, None
            elif i == 1 and second_exit:
                rgb = self.to_to_rgbs[i](x, latent, rgb)

                if return_latents:
                    return rgb, latent
                else:
                    return rgb, None

            rgb = self.to_rgbs[i](x, latent, rgb)

        if return_latents:
            return rgb, latent
        else:
            return rgb, None


class CIPSskippatchscale(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, scale_input=True, scale_input_fourier=True, scale_input_fourier_dim=512, **kwargs):
        super(CIPSskippatchscale, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = LFF(hidden_size)
        self.scale_input = scale_input
        if scale_input:
            self.lff_scale = LFF(scale_input_fourier_dim)
        # self.emb = ConstantInput(hidden_size, size=size)

        self.channels = {
            0: 512,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
            # 0: 32,
            # 1: 32,
            # 2: 32,
            # 3: 32,
            # 4: 16 * channel_multiplier,
            # 5: 8 * channel_multiplier,
            # 6: 3 * channel_multiplier,
            # 7: 2 * channel_multiplier,
            # 8: 1 * channel_multiplier,
        }

        multiplier = 1
        in_channels = int(self.channels[0])
        if not scale_input:
            self.conv1 = StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                    activation=activation)
        else:
            self.conv1 = StyledConv(int(multiplier*hidden_size+scale_input_fourier_dim), in_channels, 1, style_dim, demodulate=demodulate,
                                    activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

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

        if not self.scale_input:
            x = self.lff(coords)
        else:
            x = self.lff(coords[:, :2])
            s = self.lff_scale(coords[:, 2:])
            x = torch.cat((x, s), 1)

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
            for j in range(self.to_rgb_stride):
                x = self.linears[i*self.to_rgb_stride + j](x, latent)

            rgb = self.to_rgbs[i](x, latent, rgb)

        if return_latents:
            return rgb, latent
        else:
            return rgb, None


class CIPSskippatchmip(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, **kwargs):
        super(CIPSskippatchmip, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = LFFMip(hidden_size)
        # self.emb = ConstantInput(hidden_size, size=size)

        self.channels = {
            0: 512,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
            # 0: 32,
            # 1: 32,
            # 2: 32,
            # 3: 32,
            # 4: 16 * channel_multiplier,
            # 5: 8 * channel_multiplier,
            # 6: 3 * channel_multiplier,
            # 7: 2 * channel_multiplier,
            # 8: 1 * channel_multiplier,
        }

        multiplier = 1
        in_channels = int(self.channels[0])
        self.conv1 = StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

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
            for j in range(self.to_rgb_stride):
                x = self.linears[i*self.to_rgb_stride + j](x, latent)

            rgb = self.to_rgbs[i](x, latent, rgb)

        if return_latents:
            return rgb, latent
        else:
            return rgb, None


class CIPSskippatchpgpe(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, **kwargs):
        super(CIPSskippatchpgpe, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.hidden_size = hidden_size
        self.lff = LFF(hidden_size)
        # self.emb = ConstantInput(hidden_size, size=size)

        self.channels = {
            0: 512,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
            # 0: 32,
            # 1: 32,
            # 2: 32,
            # 3: 32,
            # 4: 16 * channel_multiplier,
            # 5: 8 * channel_multiplier,
            # 6: 3 * channel_multiplier,
            # 7: 2 * channel_multiplier,
            # 8: 1 * channel_multiplier,
        }

        multiplier = 1
        in_channels = int(self.channels[0])
        self.conv1 = StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

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

    
    def update_pe_mask(self, unmasking_ratio, device):
        # print(self.lff.ffm.conv.weight.shape) = torch.Size([512, 2, 1, 1])
        # print(self.lff.ffm.conv.bias.shape) = torch.Size([512])
        # print(self.conv1.conv.weight.shape) = torch.Size([1, 512, 512, 1, 1])

        full_dim = self.hidden_size
        unmask_dim = int(self.hidden_size * unmasking_ratio)
        mask_dim = full_dim - unmask_dim

        pe_weight_mask = torch.cat((torch.ones(unmask_dim, self.lff.ffm.conv.weight.shape[1], 1, 1), torch.zeros(mask_dim, self.lff.ffm.conv.weight.shape[1], 1, 1)), 0).to(device)
        pe_bias_mask = torch.cat((torch.ones(unmask_dim), torch.zeros(mask_dim)), 0).to(device)
        # conv1_weight_mask = torch.cat((torch.ones(1, self.conv1.conv.weight.shape[1], unmask_dim, 1, 1), torch.zeros(1, self.conv1.conv.weight.shape[1], mask_dim, 1, 1)), 2).to(device)

        self.lff.ffm.conv.weight.data.mul_(pe_weight_mask)
        self.lff.ffm.conv.bias.data.mul_(pe_bias_mask)
        # self.conv1.conv.weight.data.mul_(conv1_weight_mask)


    def lock_grad_pe_mask(self, unmasking_ratio, device):
        # print(self.lff.ffm.conv.weight.shape) = torch.Size([512, 2, 1, 1])
        # print(self.lff.ffm.conv.bias.shape) = torch.Size([512])
        # print(self.conv1.conv.weight.shape) = torch.Size([1, 512, 512, 1, 1])

        full_dim = self.hidden_size
        unmask_dim = int(self.hidden_size * unmasking_ratio)
        mask_dim = full_dim - unmask_dim

        pe_weight_mask = torch.cat((torch.zeros(unmask_dim, self.lff.ffm.conv.weight.shape[1], 1, 1), torch.ones(mask_dim, self.lff.ffm.conv.weight.shape[1], 1, 1)), 0).to(device)
        pe_bias_mask = torch.cat((torch.zeros(unmask_dim), torch.ones(mask_dim)), 0).to(device)
        # conv1_weight_mask = torch.cat((torch.zeros(1, self.conv1.conv.weight.shape[1], unmask_dim, 1, 1), torch.ones(1, self.conv1.conv.weight.shape[1], mask_dim, 1, 1)), 2).to(device)

        self.lff.ffm.conv.weight.grad.mul_(pe_weight_mask)
        self.lff.ffm.conv.bias.grad.mul_(pe_bias_mask)
        # self.conv1.conv.weight.grad.mul_(conv1_weight_mask)


    def forward(self,
                coords,
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
            for j in range(self.to_rgb_stride):
                x = self.linears[i*self.to_rgb_stride + j](x, latent)

            rgb = self.to_rgbs[i](x, latent, rgb)

        if return_latents:
            return rgb, latent
        else:
            return rgb, None