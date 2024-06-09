import math
import torch
import torch.nn as nn
import numpy as np 
import collections
from itertools import repeat
from collections import OrderedDict


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)


class SelfONNLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,q=1,mode='fast',dropout=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.q = q
        self.mode = mode # 'low_mem'
        self.dropout = nn.Dropout2d(dropout) if dropout is not None else None
        
        self.weights = nn.Parameter(torch.Tensor(out_channels,q*in_channels,*self.kernel_size)) # Q x C x K x D
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters_like_torch()
        
                
    def reset_parameters(self):
        bound = 0.01
        nn.init.uniform_(self.bias,-bound,bound)
        for q in range(self.q): nn.init.xavier_uniform_(self.weights[q])
        
    def reset_parameters_like_torch(self):
        gain = nn.init.calculate_gain('tanh')
        # gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.weights,gain=gain)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        # Input to layer
        if self.mode=='fast':
            x = torch.cat([(x**i) for i in range(1,self.q+1)],dim=1)
            if self.dropout is not None: x = self.dropout(x)
            x = torch.nn.functional.conv2d(x,self.weights,bias=self.bias,stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        
        elif self.mode == 'low_mem':
            y = x
            x = torch.nn.functional.conv2d(y,self.weights[:,:self.in_channels,:,:],bias=None,stride=self.stride,padding=self.padding,dilation=self.dilation)
            for q in range(1,self.q): 
                x += torch.nn.functional.conv2d(
                    y**(q+1) if self.dropout is None else self.dropout(y**q+1),
                    self.weights[:,(q*self.in_channels):((q+1)*self.in_channels),:,:],
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups
            )
            if self.bias is not None: x += self.bias[None,:,None,None]

        else:
            print(self.mode)
            raise NotImplementedError
        return x
