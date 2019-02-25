#!/usr/bin/env python

import sys
import json
from pathlib import Path
import numpy as np
import torch
import skimage.io

import network
import image_converter
import utils


def main():
    model_path = Path(sys.argv[1])
    setting_path = model_path.parent.parent.parent.joinpath("settings.json")
    print(f"setting: {setting_path}")
    with open(setting_path) as fp:
        settings = json.load(fp)

    if settings["use_cuda"]:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    generator = network.Generator(settings["network"]).eval().to(device)
    generator.load_state_dict(torch.load(model_path, map_location='cpu'))

    if settings["use_yuv"]:
        converter = image_converter.YUVConverter()
    else:
        converter = image_converter.RGBConverter()

    num_bases = 10
    num_div = 100
    bases = utils.create_z(num_bases, settings["network"]["z_dimension"])

    vecs = []
    for i in range(len(bases)):
        for j in range(num_div):
            val = j / num_div
            vec = utils.slerp(val, bases[i-1], bases[i])
            vecs.append(vec)
    vecs = np.stack(vecs)
    vecs = torch.from_numpy(vecs).to(device, torch.float32)

    images = []
    with torch.no_grad():
        for i in range(num_div):
            print(f"{i} / {num_div}")
            imgs = generator.forward(vecs[i*num_bases:(i+1)*num_bases], 1)
            imgs = converter.from_generator_output(imgs.cpu().numpy())
            images.append(imgs)
    images = np.vstack(images)
    images = np.moveaxis(images, 1, -1)  # BCHW to BHWC

    imdir = Path(__file__).parent.joinpath("gif")
    for i, image in enumerate(images):
        path = imdir.joinpath(f"{i}.png")
        skimage.io.imsave(path, image)


if __name__ == '__main__':
    main()
