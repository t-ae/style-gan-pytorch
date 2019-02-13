import numpy as np


def create_z(size, length):
    z = np.random.normal(0, 1, [size, length])
    return z


def create_test_z(length):
    # interpolation
    z1 = np.zeros([8, 4, length])
    z1_start = create_z(4, length)
    z1_end = create_z(4, length)
    for i in range(8):
        z1[i] = z1_start + (z1_end - z1_start) * i / 8
    z1 = z1.reshape([32, length])

    # random
    z2 = create_z(32, length)

    return z1, z2
