import numpy as np


def create_z(size, dim):
    z = np.random.normal(0, 1, [size, dim])
    return z


def create_test_z(rows, cols, dim):
    # interpolation
    z1 = np.zeros([rows, cols, dim])
    z1_start = create_z(cols, dim)
    z1_end = create_z(cols, dim)
    for i in range(rows):
        z1[i] = z1_start + (z1_end - z1_start) * i / (rows-1)
    z1 = z1.reshape([-1, dim])

    # random
    z2 = create_z(rows * cols, dim)

    return z1, z2
