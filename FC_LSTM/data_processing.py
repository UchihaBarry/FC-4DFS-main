from math import pi
from xml.etree.ElementTree import PI
from torch.autograd import Variable
import os
import torch.nn.functional as F
import torch
import scipy.io
import numpy as np
import torch.nn as nn
from option import TrainOptionParser
from os.path import join as pjoin
from scipy.io import loadmat
from utils import *
import math


def Pos_enconding(data):
    batch, feature_dim, length = data.size()
    d_model = 204
    pe = torch.zeros(length, feature_dim)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    position = (torch.arange(0, length, dtype=torch.float)).unsqueeze(1)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.transpose(0, 1).unsqueeze(0).repeat(batch, 1, 1)
    
    return pe

def gt_interpolated(gt,len):
    if len == 15:
        subset = [0,2,4,6,8,10,12,14,17,19,21,23,25,27,29]
    elif len == 20:
        subset = [0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,29]
    elif len == 25:
        subset = [0,1,3,4,5,6,7,9,10,11,12,13,14,16,17,18,19,20,22,23,24,25,26,28,29]
    else:
        subset = range(30)
    gt_interpolated = gt[:, :, subset]
    return gt_interpolated


    

# %%

def load_data(path_,args) :
    X=[]
    y=[]
    iden = []
    for sample in os.listdir(path_):
        
        X.append(sample)
       
        y.append(read_COMA_label(sample))
        iden.append(read_iden_label(sample))
    y = np.array(y)
    iden = np.array(iden)
    
    y_vec = np.zeros(shape=(len(y), args.y_dim), dtype=np.float32)
    for i, label in enumerate(y):
        y_vec[i, label] = 1
        
    return X, y, iden
  
    
def read_COMA_label(char_label):
    if 'bareteeth' in char_label:
        label=0
    elif 'cheeks_in' in char_label:
        label = 1
    elif 'eyebrow' in char_label:
        label = 2
    elif 'high_smile' in char_label:
        label = 3
    elif 'lips_back' in char_label:
        label = 4
    elif 'lips_up' in char_label:
        label = 5
    elif 'mouth_down' in char_label:
        label = 6
    elif 'mouth_extreme' in char_label:
        label = 7
    elif 'mouth_middle' in char_label:
        label = 8
    elif 'mouth_open' in char_label:
        label = 9
    elif 'mouth_side' in char_label:
        label = 10
    elif 'mouth_up' in char_label:
        label = 11
    return label

def read_iden_label(char_label):
    if 'FaceTalk_170725_00137' in char_label:
        iden_label = 0
    elif 'FaceTalk_170728_03272' in char_label:
        iden_label = 1
    elif 'FaceTalk_170731_00024' in char_label:
        iden_label = 2
    elif 'FaceTalk_170809_00138' in char_label:
        iden_label = 3
    elif 'FaceTalk_170811_03274' in char_label:
        iden_label = 4
    elif 'FaceTalk_170811_03275' in char_label:
        iden_label = 5
    elif 'FaceTalk_170904_00128' in char_label:
        iden_label = 6
    elif 'FaceTalk_170904_03276' in char_label:
        iden_label = 7
    elif 'FaceTalk_170908_03277' in char_label:
        iden_label = 8
    elif 'FaceTalk_170912_03278' in char_label:
        iden_label = 9
    elif 'FaceTalk_170913_03279' in char_label:
        iden_label = 10
    elif 'FaceTalk_170915_00223' in char_label:
        iden_label = 11
    return iden_label


def read_landmark(path_landmark):
    data_= loadmat(path_landmark)
      
    data=data_['coma_landmarks']
    
    data=np.reshape(data, [data.shape[0],data.shape[1]*data.shape[2]])
    data=torch.from_numpy(data)
    return data


def get_labels(y ,iden_flag,args):
    batch_label_rb = np.zeros(shape=(y.size()[0], args.y_dim*3),dtype = np.float32)
    exp = y
    for i in range(y.size()[0]):
       
        batch_label_rb[i,:] = y_to_rb_label(y[i] ,args)
    y = torch.from_numpy(batch_label_rb).to(args.device)
    iden_flag = flag_to_iden(iden_flag, exp,args)
    iden_label = torch.from_numpy(iden_flag).to(args.device)
    return y, iden_label


def y_to_rb_label( label,args):
    number = int(label) 
    
    one_hot = np.zeros(36)
    one_hot[number] = 1
    one_hot[number + 12] = 1
    one_hot[number + 24] = 1
    return one_hot


def flag_to_iden(batch_iden_flag, exp, args, ):
    path_iden_label = args.path_iden_label
    iden_labels = loadmat(path_iden_label)
    iden_labels = iden_labels["iden_landmarks"].reshape(12,-1)
    iden_labels = np.array(iden_labels)

    batch_iden_label = np.zeros(shape=(batch_iden_flag.size()[0],204),dtype=np.float32)
    
    for i in range(batch_iden_flag.size()[0]):
        batch_iden_label[i] = iden_labels[int(batch_iden_flag[i]) - 1][int(exp[i])]


    return batch_iden_label



def frame_interpolate(data_X):
    data_x = torch.zeros(len(data_X),30,data_X[0].size()[1])
    num_inter = 0
    for i in range(len(data_X)):
        if data_X[i].size()[0]!=30:
            data_X[i] = data_X[i].unsqueeze(1) 
            data_X[i] = data_X[i].permute(1,2,0) 
            data_X[i] = F.interpolate(data_X[i],30,mode="linear", align_corners=False)
            data_X[i] = data_X[i].permute(0,2,1)
            data_X[i] = data_X[i].squeeze()
            num_inter+=1
        data_x[i] = data_X[i]
    print("num_inter:",num_inter)
    data_x = data_x.permute(0,2,1) 

    return data_x


def concat_label(x,label,args,duplicate=1):
    x_shape = x.shape
    if duplicate < 1:
        return x
    label.repeat(1,duplicate)
    label_shape=label.shape
    
    if len(x_shape) == 2:
        return torch.cat([x,label],1)
    elif len(x_shape) == 4:
        
        label = label.reshape(x_shape[0],label_shape[-1] , 1, 1)
        
        label = label*torch.ones([x_shape[0], label_shape[-1], x_shape[2], x_shape[3]],device=args.device)
        
        return torch.cat([x, label],1)
