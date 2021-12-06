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


parser = argparse.ArgumentParser(description='PyTorch EDNN Optimization')
parser.add_argument('--data-dir', dest='data_dir',
                    help='The directory used to load data',
                    default='data/ex1', type=str)
parser.add_argument('--epochs', default=15000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='learning rate (default: 1e-4)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--l2', '--lambda2', default = 0.5, type = float,
                    metavar = 'L2', help = 'lambda2 for optimization process')
parser.add_argument('--low', '--lower-bound', default = -1.5, type = float,
                    metavar = 'low', help = 'lower bound of fixed element of parameter')
parser.add_argument('--high', '--upper-bound', default = 1.6, type = float,
                    metavar = 'high', help = 'upper bound of fixed element of parameter')
parser.add_argument('--d', '--grid-size', default = 0.1, type = float,
                    metavar = 'grid', help = 'gridsize of fixed element of parameter')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save results',
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
    
    
def bifur(p,v,alpha,b = None):
    if b is not None:
        p = torch.cat((b,p),dim=0)
    p = p.reshape(1,-1).double()
    v = v.double()
    y = model(p)
    f = eqn(p,y)
    jcb = F.jacobian(eqn, (p,y),create_graph = True)
    A = jcb[1]        
    return (torch.norm(torch.matmul(A,v))/torch.norm(v))**2 + alpha*torch.norm(f)**2


def find_bifur(b = None):
    v1 = Variable(torch.rand(D_out),requires_grad = True)    
    if b is None:
        p1 = Variable(torch.rand(D_in),requires_grad = True)
        print(p1.shape)
    else:
        dim_b = b.shape[0]
        p1 = Variable(torch.rand(D_in - dim_b),requires_grad = True)
        print(p1.shape)
    
    opt1 = torch.optim.Adam([v1], lr=args.lr)
    opt2 = torch.optim.Adam([p1], lr=args.lr) 
    
    for i in range(args.epochs):  
        l =  bifur(p1,v1,args.l2,b)
        opt1.zero_grad()
        opt2.zero_grad()               
        l.backward() 
        opt1.step()  
        opt2.step()
        if i % args.print_freq == 0:
            print(i, p1,l)     
            
    return p1.detach().clone()
        


def optimization(H,low = None, high = None, gridsize = None):      
    bf = []    
    if not low:   
        bf.append(find_bifur())
    else:
        b =  Variable(torch.from_numpy(np.arange(low,high,gridsize).astype(np.float64))).reshape(-1,1)
        for b_val in b:
            bf.append(find_bifur(b_val))
            print(b_val)
    
    bf_name = args.save_dir + '/bifur_' + str(H) + '.pt'
    torch.save(bf,bf_name)  
 


if __name__ == '__main__':
    x = torch.load(args.data_dir + "/x.pt")
    y = torch.load(args.data_dir + "/y.pt")
    
    _,D_in = x.shape
    _,D_out = y.shape
    
    H1 = [20,40,80,160,320]
    
    for H in H1:       
        model_name = args.save_dir + '/model_H_' + str(H) 
        checkpoint = torch.load(model_name,map_location=torch.device('cpu'))
        model = Net(D_in,H,D_out)
        model.load_state_dict(checkpoint['net'])
        model.eval()
        
        if args.data_dir == "data/ex1":
            optimization(H) # for example 1
        else:
            optimization(H,low = args.low, high = args.high, gridsize = args.d) 
        
        # optimization(H,low = -1.5, high = 1.6, gridsize = 0.1) # for example 2_2d
        # optimization(H,low = 0, high = 1.6, gridsize = 0.1)  # example 2_3d
        


            
            

