#!/usr/bin/env python

import sys
from pathlib import Path
import numpy as np
import data_loader
import plot_tool
import image_converter


def main(level):

    rows = 4
    cols = 4

    settings = {
        "flip": False,
        "color_shift": False,
        "rotation": True,
    }

    image_root: Path = Path(__file__).parent.joinpath("../images")
    image_paths = list(image_root.glob("**/*00.png"))
    print(f"{len(image_paths)} images")
    loader = data_loader.TrainDataLoader(image_paths, settings)

    size = 2 * 2**level
    images = next(loader.generate(rows*cols, size, size))

    # converter test
    if False:
        converter = image_converter.RGBConverter()
        images = images.transpose([0, 3, 1, 2])
        images = converter.to_train_data(images)
        images = converter.from_generator_output(images)
        images = images.transpose([0, 2, 3, 1])

    # noise test
    if False:
        noise_scale = 0.05
        noise = np.random.normal(0, 1, images.shape) * noise_scale
        images += noise
        images = np.clip(images, 0, 1)

    plot_tool.plot_images(images, rows=rows, cols=cols)


if __name__ == '__main__':
    level = int(sys.argv[1])
    main(level)
