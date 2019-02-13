#!/usr/bin/env python

import torch
from torch.autograd import Variable
import network
import torch.nn.functional as F


gen = network.SynthesisModule()
dis = network.Discriminator()

level = 3
alpha = 0

i = Variable(torch.zeros(2, network.Z_DIM))
f1 = gen.forward(i, level, 1)
d1 = dis.forward(f1, level, 1)

f2 = gen.forward(i, level + 1, 0)
f2_ = F.avg_pool2d(f2, 2)
d2 = dis.forward(f2, level + 1, 0)

print(d1 - d2)
