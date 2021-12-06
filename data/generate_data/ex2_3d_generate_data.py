#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


N = 3000    # training data size
D_in = 3   # p = [b;c;d]
D_out = 1  # x 

# generate training data


def Cubic_Coeff(b, c):
    a = 1
    D1 = b*c/6/a**2 - b**3/27/a**3
    D2 = (c/3/a - b**2/9/a**2)**3
    return 2*a*(D1 - np.sqrt(-D2))



def Cubic_Sol(b, c,d):
    a = 1
    D1 = b*c/6/a**2 - b**3/27/a**3 - d/2/a
    return -b/3/a + 2*np.cbrt(D1)


P = np.random.uniform(low = 0, high = 2, size =(N,D_in))
P1 = np.repeat(P,4,axis = 0)
for i in range(len(P1)):    
    P1[i][1] = np.random.uniform(low = 0, high = P1[i][0]**2/3)

P1[:,2] = Cubic_Coeff(P1[:,0] ,P1[:,1])

Y = Cubic_Sol(P1[:,0] ,P1[:,1],P1[:,2]).reshape(-1,1)

x = Variable(torch.from_numpy(P1.astype(np.float64)))
y = Variable(torch.from_numpy(Y.astype(np.float64)))


torch.save(x,'ex2_3d/x.pt')
torch.save(y,'ex2_3d/y.pt')

print("Data generated!")
