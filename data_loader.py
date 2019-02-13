#!/usr/bin/env python


import random
import numpy as np
from PIL import Image
import concurrent.futures as futures


def chunks(lst, chunk_size):
    num_chunks = len(lst) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        yield lst[start:start+chunk_size]


class TrainDataLoader:
    def __init__(self, paths, settings):
        self.paths = paths

        self.flip = settings["flip"]
        self.color_shift = settings["color_shift"]

    def generate(self, batch_size, width, height):
        num_workers = 5

        with futures.ThreadPoolExecutor(num_workers) as executor:
            tasks = []
            while True:
                np.random.shuffle(self.paths)
                for paths in chunks(self.paths, batch_size):
                    newtask = executor.submit(self.load_images, paths, width, height)
                    tasks.append(newtask)
                    if len(tasks) < num_workers:
                        continue
                    task = tasks.pop(0)
                    result = task.result(100)
                    if result is not None:
                        yield result

    def load_images(self, paths, width, height):

        array = np.empty([len(paths), height, width, 3])
        for i, path in enumerate(paths):
            try:
                image = Image.open(str(path))
            except:
                return None

            image = image.resize((width, height), Image.LANCZOS)
            image = np.array(image, dtype=float)
            image /= 255

            if self.flip:
                if bool(random.getrandbits(1)):
                    image = np.flip(image, 1)

            if self.color_shift:
                shift = np.random.choice([0.7, 1, 1.3])
                image = image ** shift
            array[i] = image

        return array
