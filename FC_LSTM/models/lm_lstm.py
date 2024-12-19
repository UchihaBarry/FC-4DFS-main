from torch.autograd import Variable
import os
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from option import TrainOptionParser
from os.path import join as pjoin
from loss import *
from torch.utils.tensorboard import SummaryWriter
from utils import *
from scipy.io import loadmat
import time
from data_processing import *
from time import gmtime, strftime
import math
np.random.seed(2023)



def concat_label(x,label,args,duplicate=1):
    x_shape = x.shape
    if duplicate < 1:
        return x
    label.repeat(1,duplicate)
    label_shape=label.shape
    if len(x_shape) == 2:
        return torch.cat([x,label],1)
    elif len(x_shape) == 3:
        # label = label.reshape(x_shape[0], 1, 1, label_shape[-1])
        label = label.unsqueeze(-1)

        label = label*torch.ones([x_shape[0], label_shape[-1], x_shape[2]],device=args.device)


        return torch.cat([x, label],1)
    elif len(x_shape) == 4:
       
        label = label.reshape(x_shape[0],label_shape[-1] , 1, 1)
        
        label = label*torch.ones([x_shape[0], label_shape[-1], x_shape[2], x_shape[3]],device=args.device)
        

        return torch.cat([x, label],1)
    


   
class Generator_Model(nn.Module):
    def __init__(self,args):
        super(Generator_Model, self).__init__()
        
        self.num_frame = 1
        self.size_SRVF_H = args.size_SRVF_H
        self.size_batch = args.size_batch
        self.size_kernel = args.size_kernel
        self.num_gen_channels = args.num_gen_channels
        self.num_input_channels = args.num_input_channels

        self.y_dim = args.y_dim
        self.rb_dim = args.rb_dim    
        self.num_convs = args.num_convs    
        
        self.args = args
        self.hidden = 204
        c = torch.zeros(size=[4,self.size_batch,self.hidden]).to(args.device).requires_grad_(False)
        h = torch.zeros(size=[4,self.size_batch,self.hidden]).to(args.device).requires_grad_(False)
        self.c_h = (c,h)

        
        self.layer1 = nn.Sequential(nn.Linear(in_features = self.num_frame * 204 + (36),out_features = self.num_gen_channels * int(math.ceil(self.num_frame/2))))
        self.layer2 = self.deconv_layer(in_channels =int(self.num_gen_channels * 2**(0)+(36)),out_channels=int(self.num_gen_channels * 2**(1)))
        self.layer3 = self.deconv_layer(in_channels =int(self.num_gen_channels * 2**(1)+(36)),out_channels=int(self.num_gen_channels * 2**(2)))
        self.layer4 = self.deconv_layer(in_channels = int(self.num_gen_channels*4+(36)),out_channels = int(self.num_gen_channels*2))
        self.layer5 = self.deconv_layer(in_channels = int(self.num_gen_channels*2+(36)),out_channels = self.num_gen_channels)
        self.layer6 = self.last_conv_layer(in_channels = int(self.num_gen_channels+(36)),out_channels = 204)
        self.emb_dim = 3
        self.cross_atten = nn.MultiheadAttention(embed_dim=self.emb_dim, num_heads=1, dropout=0.2,batch_first=True)
        self.lstm = nn.LSTM(input_size=204, hidden_size=self.hidden, num_layers=4, batch_first=True)
        self.scaling_factor = nn.Parameter(torch.tensor(1.0))
        self.output = None

    
    def conv_layer(self, in_channels , out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            layer = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels , out_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.ReLU(True)
            )
            return layer
        
    def deconv_layer(self, in_channels , out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            layer = torch.nn.Sequential(
                torch.nn.ConvTranspose1d(in_channels , out_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.ReLU(True)
               
            )
            return layer
    def last_conv_layer(self, in_channels , out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            layer = torch.nn.Sequential(
                torch.nn.ConvTranspose1d(in_channels , out_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
                torch.nn.Tanh()
                
            )
            return layer
    def init_c_h(self, input):
        size_batch = input.size()[0]
        c = torch.zeros(size=[4,size_batch,self.hidden]).to(self.args.device).requires_grad_(False)
        h = torch.zeros(size=[4,size_batch,self.hidden]).to(self.args.device).requires_grad_(False)
        self.c_h = (h,c)

    def forward(self, label_y, label_exp,input ,pos, prev = None,):
        input = input.reshape(label_y.size()[0],-1)
        label = label_y.reshape(-1,36)
        input = torch.cat([input, label],1)     
        layer = self.layer1
        input = layer(input)
        input = torch.reshape(input,[label.size()[0], self.num_gen_channels, int(math.ceil(self.num_frame/2))])
        layer = nn.ReLU(inplace=True)
        
        input = layer(input)
        
        input = concat_label(input, label,args=self.args)
        
        for i in range(self.num_convs):
            name = "Gen Conv_0" + str(i)
            if i == 0:
                layer = self.layer2
            else:
                layer = self.layer3
            
            input = layer(input)
           
            input = concat_label(input, label,args=self.args)
           
        name = 'G_deconv' + str(i + 1)
        
        input = F.interpolate(input, self.num_frame)
        
        
        layer = self.layer4   
        input = layer(input)
        
        input = concat_label(input, label,args=self.args)
       
        layer = self.layer5
        input = layer(input)

    
        input = concat_label(input, label,args=self.args)
        layer = self.layer6
        input = layer(input)
        
        
        layer = self.lstm
        

       
        input = input + self.scaling_factor * pos
        input = input.permute(0,2,1)
       
        output, (c, h) = layer(input, self.c_h)
       
        self.c_h = (c, h)
       
        output = output.permute(0,2,1)
        loss_expr = F.l1_loss(label_y, label_y)
        


        return output, loss_expr

