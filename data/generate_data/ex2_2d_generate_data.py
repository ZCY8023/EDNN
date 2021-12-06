#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
N = 5000    # training data size
D_in = 2   # p = [b;c]
D_out = 1  # x 

# generate training data
# fix a = 1
def Quadratic_Sol(b, c):
    return (-b+np.sqrt(b**2-4*c))/2


P = np.random.uniform(low = -2, high = 2, size =(N,D_in))
P1 = np.repeat(P,5,axis = 0)
for i in range(len(P1)):
    P1[i][1] = np.random.uniform(low = P2[i][0]**2/8, high = P2[i][0]**2/4)


Y = Quadratic_Sol(P1[:,0],P1[:,1]).reshape(-1,1)
x = Variable(torch.from_numpy(P1.astype(np.float64)))
y = Variable(torch.from_numpy(Y.astype(np.float64)))


torch.save(x,'ex2_2d/x.pt')
torch.save(y,'ex2_2d/y.pt')
print("Data generated!")
