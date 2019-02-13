#!/usr/bin/env python

import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
import utils
import plot_tool
import network

np.random.seed(42)


def random():
    rows = 4
    cols = 4

    z = utils.create_z(rows * cols, network.Z_DIM)

    generate_and_plot(z, rows, cols, False)


def interpolate():
    rows = 8
    cols = 8

    z00 = utils.create_z(1, network.Z_DIM)
    z01 = utils.create_z(1, network.Z_DIM)
    z10 = utils.create_z(1, network.Z_DIM)
    z11 = utils.create_z(1, network.Z_DIM)

    z0 = []
    z1 = []
    for c in range(rows):
        rate = c / (rows-1)
        z0.append((rate-1)*z00 + rate*z01)
        z1.append((rate-1)*z10 + rate*z11)
    z0 = np.vstack(z0)
    z1 = np.vstack(z1)

    zs = []
    for c in range(cols):
        rate = c / (cols-1)
        zs.append((rate-1)*z0 + rate*z1)

    z = np.stack(zs, 1)
    cols = z.shape[1]
    z = z.reshape(-1, network.Z_DIM)

    generate_and_plot(z, rows, cols, False)


def generate_and_plot(z, rows, cols, show_label, alpha=1):
    z = Variable(torch.from_numpy(z).float(), volatile=True)

    generator = network.SynthesisModule().eval()
    discriminator = network.Discriminator().eval()

    if len(sys.argv) > 1:
        epoch = int(sys.argv[1])
        generator.load_state_dict(torch.load(f"weights/{epoch}_gen.pth"))
        discriminator.load_state_dict(torch.load(f"weights/{epoch}_disc.pth"))
    else:
        generator.load_state_dict(torch.load(f"weights/gen.pth"))
        discriminator.load_state_dict(torch.load(f"weights/disc.pth"))

    images = generator.forward(z, alpha)
    if show_label:
        scores = discriminator.forward(images, alpha)
        scores = scores.data.numpy()
    else:
        scores = None

    images = np.clip((images.data.numpy() + 1) / 2, 0, 1)
    images = images.transpose([0, 2, 3, 1])

    plot_tool.plot_images(images, scores, rows=rows, cols=cols)


if __name__ == '__main__':
    random()
    # interpolate()
