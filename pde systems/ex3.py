#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd.functional as F
import numpy as np
import scipy.io as sio   # load .mat file


exact_bifur =  sio.loadmat('data/ex3/exact_bifur.mat')
exact_bifur = exact_bifur['exact_bifur']

branch = [1,2,3,4]

for r in range(len(branch)):
    br = branch[r]
    train = sio.loadmat('data/ex3/ex3.mat')
    train = train['sol'][0]
    train = train[br-1]
    N,_ = train.shape  # size of loaded data: sol + parameter
    D_in = 1   # size of parameter 
    D_out = N - 1  # size of sol
    
    P = train[-1].reshape(-1,1)
    Y = train[:-1].transpose()
    x = Variable(torch.from_numpy(P.astype(np.float64)))
    y = Variable(torch.from_numpy(Y.astype(np.float64)))



def pde(x,output):
    h = 1/N
    U = torch.transpose(output,-1,0)
    x = x.reshape(1,-1)
    f = Variable(torch.zeros(U.size()))
    for i in range(1,D_out-1):
        f[i] = (U[i+1] - 2*U[i] + U[i-1])/(h**2) - (U[i]**2)*(U[i]**2 - x)      
    f[0] = (U[1] - 2*U[0] + U[0])/(h**2) - (U[0]**2)*(U[0]**2 - x) 
    f[D_out-1] = (0 - 2*U[D_out-1] + U[D_out-2])/(h**2) - (U[D_out-1]**2)*(U[D_out-1]**2 - x) 
    return f