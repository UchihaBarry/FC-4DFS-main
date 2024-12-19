## Code modified from https://github.com/gbouritsas/Neural3DMM

from torch.utils.data import Dataset
import torch
import numpy as np
import os
from scipy.io import loadmat
import random


class autoencoder_dataset(Dataset):

    def __init__(self, template, neutral_root_dir, points_dataset, normalization = True, dummy_node = True):
        

        
        self.normalization = normalization
        #self.root_dir = root_dir
        self.neutral_root_dir = neutral_root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(neutral_root_dir, points_dataset,'vertices.npy')) 
        self.template=template
        self.paths_lands=np.load(os.path.join(neutral_root_dir, points_dataset, 'landmarks.npy'))

    def __len__(self):
        return len(self.paths)
    
    

    def __getitem__(self, idx):
        basename = self.paths[idx] 
        basename_landmarks=self.paths_lands[idx]
        # print(os.path.join(self.neutral_root_dir,'points_'+self.points_dataset , basename+'.npy'))

        verts_input= np.load(os.path.join(self.neutral_root_dir, self.points_dataset,'points_target', basename+'.npy'), allow_pickle=True)
        # 加载中性mesh
        if os.path.isfile(os.path.join(self.neutral_root_dir, self.points_dataset, 'landmarks_neutral', basename + '.npy')):
           landmarks_neutral = np.load(os.path.join(self.neutral_root_dir, self.points_dataset, 'landmarks_neutral', basename + '.npy'),allow_pickle=True)
        else:
            landmarks_neutral=np.zeros(np.shape(verts_input))
            
        if os.path.isfile(os.path.join(self.neutral_root_dir, self.points_dataset, 'points_neutral', basename + '.npy')):
           verts_neutral = np.load(os.path.join(self.neutral_root_dir, self.points_dataset, 'points_neutral', basename + '.npy'),allow_pickle=True)
        else:
            verts_neutral=np.zeros(np.shape(verts_input))


        
        landmarks=np.load(os.path.join(self.neutral_root_dir, self.points_dataset,'landmarks_target', basename_landmarks+'.npy'), allow_pickle=True)
        landmarks=landmarks# -landmarks_neutral
        


        verts_input[np.where(np.isnan(verts_input))]=0.0

        
        verts_input = verts_input.astype('float32')

        landmarks=landmarks.astype('float32')

        verts_neutral = verts_neutral.astype('float32')
        
        landmarks_neutral = landmarks_neutral.astype('float32')

        if self.dummy_node:

            verts_ = np.zeros((verts_input.shape[0] + 1, verts_input.shape[1]), dtype=np.float32)
            verts_t = np.zeros((verts_input.shape[0] + 1, verts_input.shape[1]), dtype=np.float32)
            verts_[:-1,:] = verts_input
            verts_t[:-1,:] = verts_neutral
            verts_input=verts_
            verts_neutral = verts_t

        
    
        verts_input = torch.Tensor(verts_input)
        landmarks = torch.Tensor(landmarks)
        verts_neutral = torch.Tensor(verts_neutral)
        landmarks_neutral = torch.Tensor(landmarks_neutral)
       
        
        sample = {'points': verts_input, 'landmarks': landmarks,'neutral_lms':landmarks_neutral,'neutral_points':verts_neutral}# 'neurals':verts_neural
        
        return sample

