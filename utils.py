import numpy as np


def slerp(val, low, high):
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


def create_z(size, dim):
    z = np.random.normal(0, 1, [size, dim])
    return z


def create_test_z(rows, cols, dim):
    # interpolation
    z1 = np.zeros([rows, cols, dim])
    z1_start = create_z(cols, dim)
    z1_end = create_z(cols, dim)
    for i in range(rows):
        val = i / (rows-1)
        for j in range(cols):
            z1[i, j] = slerp(val, z1_start[j], z1_end[j])
    z1 = z1.reshape([-1, dim])

    # random
    z2 = create_z(rows * cols, dim)

    return z1, z2
