import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class PixelNormalizationLayer(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.use_unit_length = settings["pixel_norm_unit_length"]
        self.epsilon = settings["epsilon"]

    def forward(self, x):
        # x is [B, C, H, W]
        x2 = x ** 2

        length_inv = torch.rsqrt(x2.sum(1, keepdim=True) + self.epsilon)

        if self.use_unit_length:
            # unit length
            return x * length_inv
        else:
            # original code
            return np.sqrt(x.size()[1]) * x * length_inv


class MinibatchStdConcatLayer(nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.num_concat = settings["std_concat"]["num_concat"]
        self.group_size = settings["std_concat"]["group_size"]
        self.use_variance = settings["std_concat"]["use_variance"]  # else use variance
        self.epsilon = settings["epsilon"]

    def forward(self, x):
        if self.num_concat == 0:
            return x

        group_size = self.group_size
        # x is [B, C, H, W]
        size = x.size()
        assert(size[0] % group_size == 0)
        M = size[0]//group_size

        x32 = x.to(torch.float32)

        y = x32.view(group_size, M, -1)  # [group_size, M, -1]
        mean = y.mean(0, keepdim=True)  # [1, M, -1]
        y = ((y - mean)**2).mean(0)  # [M, -1]
        if not self.use_variance:
            y = (y + self.epsilon).sqrt()
        y = y.mean(1)  # [M]
        y = y.repeat(group_size, 1)  # [group_size, M]
        y = y.view(-1, 1, 1, 1)
        y1 = y.expand([size[0], 1, size[2], size[3]])
        y1 = y1.to(x.dtype)

        if self.num_concat == 1:
            return torch.cat([x, y1], 1)

        # self.num_concat == 2
        y = x32.view(M, group_size, -1)  # [M, group_size, -1]
        mean = y.mean(1, keepdim=True)  # [M, 1, -1]
        y = ((y - mean) ** 2).mean(1)  # [M, -1]
        if self.use_variance:
            y = (y + 1e-8).sqrt()
        y = y.mean(1, keepdim=True)  # [M, 1]
        y = y.repeat(1, group_size)  # [M, group_size]
        y = y.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        y2 = y.expand([size[0], 1, size[2], size[3]])
        y2 = y2.to(x.dtype)

        return torch.cat([x, y1, y2], 1)


class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, gain=np.sqrt(2)):
        super().__init__()
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        init.normal_(weight)
        self.weight = nn.Parameter(weight)
        scale = gain / np.sqrt(in_channels * kernel_size * kernel_size)
        self.register_buffer("scale", torch.tensor(scale))

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        scaled_weight = self.weight * self.scale
        return F.conv2d(x, scaled_weight, self.bias, self.stride, self.padding)


class WSConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, gain=np.sqrt(2)):
        super().__init__()
        weight = torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        init.normal_(weight)
        self.weight = nn.Parameter(weight)
        scale = gain / np.sqrt(in_channels * kernel_size * kernel_size)
        self.register_buffer("scale", torch.tensor(scale))

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        scaled_weight = self.weight * self.scale
        return F.conv_transpose2d(x, scaled_weight, self.bias, self.stride, self.padding)


class AdaIN(nn.Module):
    def __init__(self, dim, w_dim, epsilon):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.transform = WSConv2d(w_dim, dim*2, 1, 1, 0)

    def forward(self, x, w):
        # Instance norm
        x_mean = x.mean((2, 3), keepdim=True)
        x = x - x_mean
        x_mean2 = (x**2).mean((2, 3), keepdim=True)
        x = x * torch.rsqrt(x_mean2 + self.epsilon)

        # scale
        style = self.transform(w).view([-1, 2, self.dim, 1, 1])
        scale = style[:, 0] + 1
        bias = style[:, 1]

        return scale * x + bias


class NoiseLayer(nn.Module):
    def __init__(self, dim, size):
        super().__init__()

        self.fixed = False

        self.size = size
        self.register_buffer("fixed_noise", torch.randn([1, 1, size, size]))

        self.noise_scale = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        batch_size = x.size()[0]
        if self.fixed:
            noise = self.fixed_noise.expand(batch_size, -1, -1, -1)
        else:
            noise = torch.randn([batch_size, 1, self.size, self.size]).to(x.device, x.dtype)

        return x + noise * self.noise_scale


class LatentTransformation(nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.z_dim = settings["z_dimension"]
        self.w_dim = settings["w_dimension"]
        self.latent_normalization = PixelNormalizationLayer(settings) if settings["normalize_latents"] else None
        activation = nn.LeakyReLU(negative_slope=0.2)

        # outputs [network_dim*2, 1, 1]
        self.latent_transform = nn.Sequential(
            WSConv2d(self.z_dim, self.z_dim, 1, 1, 0),
            activation,
            WSConv2d(self.z_dim, self.z_dim, 1, 1, 0),
            activation,
            WSConv2d(self.z_dim, self.z_dim, 1, 1, 0),
            activation,
            WSConv2d(self.z_dim, self.z_dim, 1, 1, 0),
            activation,
            WSConv2d(self.z_dim, self.z_dim, 1, 1, 0),
            activation,
            WSConv2d(self.z_dim, self.z_dim, 1, 1, 0),
            activation,
            WSConv2d(self.z_dim, self.w_dim, 1, 1, 0),
            activation
        )

    def forward(self, latent):
        latent = latent.view([-1, self.z_dim, 1, 1])

        if self.latent_normalization is not None:
            latent = self.latent_normalization(latent)

        return self.latent_transform(latent)


class SynthFirstBlock(nn.Module):
    def __init__(self, start_dim, output_dim, w_dim, epsilon):
        super().__init__()

        self.base_image = nn.Parameter(torch.ones(1, start_dim, 4, 4))

        self.conv = WSConv2d(start_dim, output_dim, 3, 1, 1)

        self.noise1 = NoiseLayer(start_dim, 4)
        self.noise2 = NoiseLayer(output_dim, 4)

        self.adain1 = AdaIN(start_dim, w_dim, epsilon)
        self.adain2 = AdaIN(output_dim, w_dim, epsilon)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, w):
        batch_size = w.size()[0]

        x = self.base_image.expand(batch_size, -1, -1, -1)
        x = self.noise1(x)
        x = self.activation(x)
        x = self.adain1(x, w)

        x = self.conv(x)
        x = self.noise2(x)
        x = self.activation(x)
        x = self.adain2(x, w)

        return x


class SynthBlock(nn.Module):
    def __init__(self, input_dim, output_dim, output_size, w_dim, epsilon):
        super().__init__()

        self.conv1 = WSConv2d(input_dim, output_dim, 3, 1, 1)
        self.conv2 = WSConv2d(output_dim, output_dim, 3, 1, 1)

        self.noise1 = NoiseLayer(output_dim, output_size)
        self.noise2 = NoiseLayer(output_dim, output_size)

        self.adain1 = AdaIN(output_dim, w_dim, epsilon)
        self.adain2 = AdaIN(output_dim, w_dim, epsilon)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, w):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        x = self.conv1(x)
        x = self.noise1(x)
        x = self.activation(x)
        x = self.adain1(x, w)

        x = self.conv2(x)
        x = self.noise2(x)
        x = self.activation(x)
        x = self.adain2(x, w)

        return x


class SynthesisModule(nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.w_dim = settings["w_dimension"]

        epsilon = settings["epsilon"]

        self.blocks = nn.ModuleList([
            SynthFirstBlock(512, 512, self.w_dim, epsilon),
            SynthBlock(512, 512, 8, self.w_dim, epsilon),
            SynthBlock(512, 256, 16, self.w_dim, epsilon),
            SynthBlock(256, 128, 32, self.w_dim, epsilon),
            SynthBlock(128, 64, 64, self.w_dim, epsilon),
            SynthBlock(64, 32, 128, self.w_dim, epsilon),
            SynthBlock(32, 16, 256, self.w_dim, epsilon)
        ])

        self.to_rgbs = nn.ModuleList([
            WSConv2d(512, 3, 1, 1, 0, gain=1),
            WSConv2d(512, 3, 1, 1, 0, gain=1),
            WSConv2d(256, 3, 1, 1, 0, gain=1),
            WSConv2d(128, 3, 1, 1, 0, gain=1),
            WSConv2d(64, 3, 1, 1, 0, gain=1),
            WSConv2d(32, 3, 1, 1, 0, gain=1),
            WSConv2d(16, 3, 1, 1, 0, gain=1)
        ])

        self.register_buffer("level", torch.tensor(1, dtype=torch.int32))

    def set_noise_fixed(self, fixed):
        for module in self.modules():
            if module is NoiseLayer:
                module.fixed = fixed

    def forward(self, w, alpha):
        # w is [batch_size. level, w_dim, 1, 1]
        level = self.level.item()

        x = self.blocks[0](w[:, 0])

        if level == 1:
            x = self.to_rgbs[0](x)
            return x

        for i in range(1, level-1):
            x = self.blocks[i](x, w[:, i])

        x2 = x
        x2 = self.blocks[level-1](x2, w[:, level-1])
        x2 = self.to_rgbs[level-1](x2)

        if alpha == 1:
            x = x2
        else:
            x1 = self.to_rgbs[level - 2](x)
            x1 = F.interpolate(x1, scale_factor=2, mode="bilinear")
            x = torch.lerp(x1, x2, alpha)

        return x


class Generator(nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.latent_transform = LatentTransformation(settings)
        self.synthesis_module = SynthesisModule(settings)
        self.style_mixing_prob = settings["style_mixing_prob"]

    def set_level(self, level):
        self.synthesis_module.level.fill_(level)

    def forward(self, z, alpha):
        batch_size = z.size()[0]
        level = self.synthesis_module.level.item()

        # w is [B, level, z_dim, 1, 1]
        w = self.latent_transform(z)\
            .view(batch_size, 1, -1, 1, 1)\
            .expand(-1, level, -1, -1, -1)

        # style mixing
        if self.training and level >= 2:
            z_mix = torch.randn_like(z)
            w_mix = self.latent_transform(z_mix)
            for batch_index in range(batch_size):
                if np.random.uniform(0, 1) < self.style_mixing_prob:
                    cross_point = np.random.randint(1, level)
                    w[batch_index, cross_point:] = w_mix[batch_index]

        fakes = self.synthesis_module(w, alpha)

        return fakes


class DBlock(nn.Module):
    def __init__(self, inpit_dim, output_dim):
        super().__init__()

        self.conv1 = WSConv2d(inpit_dim, output_dim, 3, 1, 1)
        self.conv2 = WSConv2d(output_dim, output_dim, 3, 1, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, 2)
        return x


class DLastBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.conv1 = WSConv2d(input_dim, input_dim, 3, 1, 1)
        self.conv2 = WSConv2d(input_dim, input_dim, 4, 1, 0)
        self.conv3 = WSConv2d(input_dim, 1, 3, 1, 1, gain=1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.from_rgbs = nn.ModuleList([
            WSConv2d(3, 16, 1, 1, 0),
            WSConv2d(3, 32, 1, 1, 0),
            WSConv2d(3, 64, 1, 1, 0),
            WSConv2d(3, 128, 1, 1, 0),
            WSConv2d(3, 256, 1, 1, 0),
            WSConv2d(3, 512, 1, 1, 0),
            WSConv2d(3, 512, 1, 1, 0)
        ])

        self.blocks = nn.ModuleList([
            DBlock(16, 32),
            DBlock(32, 64),
            DBlock(64, 128),
            DBlock(128, 256),
            DBlock(256, 512),
            DBlock(512, 512),
            DLastBlock(512)
        ])

        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.register_buffer("level", torch.tensor(1, dtype=torch.int32))

    def set_level(self, level):
        self.level.fill_(level)

    def forward(self, x, alpha):
        level = self.level.item()

        if level == 1:
            x = self.from_rgbs[-1](x)
            x = self.activation(x)
            x = self.minibatch_std_concat(x)
            x = self.blocks[-1](x)
        else:
            x2 = self.from_rgbs[-level](x)
            x2 = self.activation(x2)
            x2 = self.blocks[-level](x2)

            if alpha == 1:
                x = x2
            else:
                x1 = F.avg_pool2d(x, 2)
                x1 = self.from_rgbs[-level+1](x1)
                x1 = self.activation(x1)

                x = torch.lerp(x1, x2, alpha)

            for l in range(1, level-1):
                x = self.blocks[-level+l](x)

            x = self.blocks[-1](x)

        return x.view([-1, 1])
