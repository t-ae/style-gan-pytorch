import numpy as np


class RGBConverter:
    def to_train_data(self, images):
        # images is in [0, 1]
        return images * 2 - 1

    def from_generator_output(self, images):
        # images is in (about) [-1, 1]
        return np.clip((images + 1) / 2, 0, 1)


class YUVConverter:
    # ITU-R BT.601
    # YCbCr
    def to_train_data(self, images):
        # images is in [0, 1]
        yuv = np.zeros_like(images, dtype=np.float)
        yuv[:, 0] = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        yuv[:, 1] = -0.168736 * images[:, 0] - 0.331264 * images[:, 1] + 0.5 * images[:, 2]
        yuv[:, 2] = 0.5 * images[:, 0] - 0.418688 * images[:, 1] - 0.081312 * images[:, 2]
        yuv[:, 0] = yuv[:, 0]*2 - 1
        yuv[:, 1:] *= 2
        return yuv

    def from_generator_output(self, images):
        # images is in [-1, 1]
        images = images.copy()
        images = np.clip(images, -1, 1)
        images[:, 0] = (images[:, 0] + 1)/2
        images[:, 1:] /= 2

        rgb = np.zeros_like(images, dtype=np.float)
        rgb[:, 0] = images[:, 0] + 1.402 * images[:, 2]
        rgb[:, 1] = images[:, 0] - 0.344136 * images[:, 1] - 0.714136 * images[:, 2]
        rgb[:, 2] = images[:, 0] + 1.772 * images[:, 1]

        return np.clip(rgb, 0, 1)
