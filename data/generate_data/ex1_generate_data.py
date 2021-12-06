#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

N = 1600    # batch size  
D_in = 1   # p
D_out = 1  # x 

# x^2 = p, choose the positive branch with x = sqrt(p)
# generate training data
def Sol(p):
    return np.sqrt(p)

P = np.random.uniform(low = 0, high = 2, size =(N,1))
Y = Sol(P)
x = Variable(torch.from_numpy(P.astype(np.float64)))
y = Variable(torch.from_numpy(Y.astype(np.float64)))

torch.save(x,'ex1/x.pt')
torch.save(y,'ex1/y.pt')
print("Data generated!")
