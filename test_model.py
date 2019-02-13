#!/usr/bin/env python

import json
import torch
import torch.nn.functional as F
import network
import utils

# parameters
SETTING_JSON_PATH = "./settings.json"


def test_up_down_sample():
    with open(SETTING_JSON_PATH) as fp:
        settings = json.load(fp)
    gen = network.Generator(settings["network"])
    gen.style_mixing_prob = 0
    disc = network.Discriminator(settings["network"])

    level = 6

    gen.set_level(level)
    disc.set_level(level)

    z = utils.create_z(16, settings["network"]["z_dimension"])
    z = torch.from_numpy(z).float()

    out_g = gen.forward(z, 1)
    out_d = disc.forward(out_g, 1)

    gen.set_level(level+1)
    disc.set_level(level+1)

    out_g2 = gen.forward(z, 1e-8)
    out_d2 = disc.forward(out_g2, 1e-8)

    d1d2 = out_d - out_d2

    print(F.interpolate(out_g, scale_factor=2, mode="bilinear") - out_g2)
    print(d1d2)


if __name__ == '__main__':
    test_up_down_sample()
