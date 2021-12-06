#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd.functional as F
import numpy as np
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='PyTorch EDNN Training')
parser.add_argument('--data-dir', dest='data_dir',
                    help='The directory used to load data',
                    default='data/ex1', type=str)
parser.add_argument('--epochs', default=30000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=5*1e-4, type=float,
                    metavar='LR', help='learning rate (default: 5*1e-4)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--l1', '--lambda1', default = 0.5, type = float,
                    metavar = 'L1', help = 'lambda1 for training process')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='model_result/ex1', type=str)


class Net(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(D_in, H).double()
        self.fc2 = nn.Linear(H, D_out).double()
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        y_predict = self.fc2(x)
        return y_predict

global args
args = parser.parse_args()


def eqn(x,output):
    if args.data_dir == "data/ex1":
        return output**2 - x   
    elif args.data_dir == "data/ex2_2d":
        output = output.reshape(1,-1)
        return output**2 + x[:,0]*output + x[:,1]
    elif args.data_dir == "data/ex2_3d":
        output = output.reshape(1,-1)
        return output**3 + x[:,0]*output**2 + x[:,1]*output + x[:,2]

# loss function f1
def my_loss(x,output,target,alpha):
    loss1 = torch.mean((output - target)**2)
    loss2 = torch.mean(eqn(x,output)**2)
    return loss1 + alpha*loss2


def train(H):    
    model = Net(D_in,H,D_out)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ls = []    
    for i in tqdm(range(args.epochs)):
        y_pred = model(x)   
        loss = my_loss(x,y_pred, y, args.l1)
        ls.append(loss.item())
        if i % args.print_freq == 0:
            print(i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':args.epochs,'alpha':args.l1}
    model_name = args.save_dir + '/model_H_' + str(H) 
    torch.save(state,model_name)
    loss_name = args.save_dir + '/loss1_H_' + str(H) + '.pt'
    torch.save(ls,loss_name)
        
 

if __name__ == '__main__':
    x = torch.load(args.data_dir + "/x.pt")
    y = torch.load(args.data_dir + "/y.pt")
    
    _,D_in = x.shape
    _,D_out = y.shape
    
    H1 = [20,40,80,160,320]
    
    for H in H1:        
        train(H)

    