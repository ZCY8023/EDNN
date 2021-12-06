#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd.functional as F
import numpy as np
import scipy.io as sio   # load .mat file


train = sio.loadmat('data/ex4/ex5_a=1:3.mat')
exact = torch.load('data/ex4/exact')
train = train['U1']
N,_ = train.shape  # size of loaded data: sol + parameter
D_in = 4  # size of parameter 
D_out = N - 4  # size of sol
var_size = D_out // 2

train = train.transpose()

P = train[:,-D_in:]  # P = (a,b,mu,d)
Y = train[:,:-D_in]
x = Variable(torch.from_numpy(P.astype(np.float64)))
y = Variable(torch.from_numpy(Y.astype(np.float64)))
# x = x.cuda()
# y = y.cuda()


def eqn(x,output):
    h = 1/(var_size - 1)
    U = torch.transpose(output,-1,0)    #(u,v)
    u = U[:var_size]
    v = U[var_size:]
   
    x = torch.transpose(x,-1,0)
    a = x[0]
    b = x[1]
    mu = x[2]
    d = x[-1]
    
    f = Variable(torch.zeros(U.size()))
    for i in range(1,var_size-1):
        f[i] = (u[i+1] - 2*u[i] + u[i-1])/(h**2) + mu*(a - u[i] + (u[i]**2)*v[i])
        f[i+var_size] = d*(v[i+1] - 2*v[i] + v[i-1])/(h**2) + mu*(b - (u[i]**2)*v[i])
    f[0] = (u[1] - 2*u[0] + u[1])/(h**2) + mu*(a - u[0] + (u[0]**2)*v[0])
    f[var_size-1] = (u[-2] - 2*u[-1] + u[-2])/(h**2) + mu*(a - u[-1] + (u[-1]**2)*v[-1])
    f[var_size] = d*(v[1] - 2*v[0] + v[1])/(h**2) + mu*(b - (u[0]**2)*v[0])
    f[-1] = d*(v[-2] - 2*v[-1] + v[-2])/(h**2) + mu*(b - (u[-1]**2)*v[-1])   
    return f


