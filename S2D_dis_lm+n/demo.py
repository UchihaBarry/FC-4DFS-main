import numpy as np
import json
import os
import copy
import _pickle as pickle
from scipy.io import loadmat

import mesh_sampling
import trimesh
from shape_data import ShapeData
from autoencoder_dataset import autoencoder_dataset
from torch.utils.data import DataLoader
from spiral_utils import get_adj_trigs, generate_spirals

import torch
from sklearn.metrics.pairwise import euclidean_distances
from save_meshes import save_meshes
from PIL import Image
import glob
import argparse
####################


parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('--test_fold', type=int, help='Change the fold here from 1 to 4', default=1)
parser.add_argument('--Split', type=str, help='Expressions split protocol: Split=Expr, Id split: Split=Id', default='Id')
parser.add_argument('--Lands', type=str, help='Lands=GT if you want to reproduce table 1 results, Lands=Motion3DGAN if you want to show Motion3DGAN generated samples', default='GT')
parser.add_argument('--label', type=str, help='the desired label for Motion3DGAN samples', default='0')
parser.add_argument('--checkpoint_file', type=str, help='the desired label for Motion3DGAN samples', default='checkpoint')
# parser.add_argument('--save_dir', type=str, help='the desired label for Motion3DGAN samples', default='0')
parser.add_argument('--sample_dir', type=str, help='the desired label for Motion3DGAN samples', default='../../gan_/same_z')
parser.add_argument('--level', type=str, help='the desired label for Motion3DGAN samples', default='2')
parser.add_argument('--save_path_Meshes', type=str, help='the desired label for Motion3DGAN samples', default='../../gan_/gan_1d_vae_4deconv_rb_single_label_norm_/test_data_mesh/')
parser.add_argument('--num', type=int, help='the desired label for Motion3DGAN samples', default=20)
parser.add_argument('--device', type=int, help='the desired label for Motion3DGAN samples', default=0)
args_ = parser.parse_args()

from model.models_double import SpiralAutoencoder



GPU = True
device_idx = args_.device
torch.cuda.get_device_name(device_idx)



root_dir = '../../code/neural3dmm/dataset/CoMA/preprocessed/'
results_dir = './Models/train'
testresults_dir = './Results/tst'
model_path = './Models/train/latent_16/checkpoints'

sample_dir = args_.sample_dir
if sample_dir == '../gan_1d/generate_sample/same_z':
    num_sample = 1
elif sample_dir == '../gan_1d_/generate_sample/same_z':
    num_sample = 2
else:
    num_sample = 2


filter_sizes_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
args = { 'neutral_data': os.path.join(root_dir),
        'results_folder': os.path.join(results_dir),
        'testresults_folder': os.path.join(testresults_dir),
        'save_path_animations': './Results/Gifs/',
        'seed': 2, 'loss': 'l1',
        'batch_size': 16, 'num_epochs': 300, 'eval_frequency': 200, 'num_workers': 4,
        'filter_sizes_enc': filter_sizes_enc, 'filter_sizes_dec': filter_sizes_dec,
        'nz': 16,
        'resume': False, 'nbr_landmarks': 68,}

if __name__ == "__main__":
    print("#########################\n\n\t loading params \n\n########################")
    sample = torch.load('./sampling.pth')
    sizes = sample['sizes']
    spiral_sizes = sample['spiral_size']
    tspirals = sample['spirals']
    tD = sample['D']
    tU = sample['U']
    if GPU:
        device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model = SpiralAutoencoder(filters_enc=args['filter_sizes_enc'],
                              filters_dec=args['filter_sizes_dec'],
                              latent_size=args['nz'],
                              sizes=sizes,
                              nbr_landmarks=args['nbr_landmarks'],
                              spiral_sizes=spiral_sizes,
                              spirals=tspirals,
                              D=tD, U=tU, device=device).to(device)
    print('loading checkpoint from file %s'%(os.path.join(model_path,args_.checkpoint_file+'.pth.tar')))
    checkpoint_dict = torch.load(os.path.join(model_path,args_.checkpoint_file+'.pth.tar'),map_location=device)
    model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
num_mesh = args_.num
for exp in range(12):
    for iden in range(9,12):
            
            sample_path = sample_dir + '/level'+args_.level+'/exp_'+str(exp)+'_iden_'+str(iden)+'.npy'

            print("sample path:",sample_path)
            input_lms = np.load(sample_path)
            input_lms = input_lms.astype(np.float32)
            input_lms = torch.from_numpy(input_lms).to(device)
            
            
            path_iden_label = "../S2D_Data_norm/iden_landmarks.mat"
            iden_labels = loadmat(path_iden_label)
            iden_labels = iden_labels["iden_landmarks"].reshape(12,-1)
            iden_labels = np.array(iden_labels,dtype=np.float32)
            neutral_lms = iden_labels[iden] #exp
            # print(neutral_lms)
            neutral_lms = neutral_lms.reshape(1, 68, 3)
            neutral_lms = torch.from_numpy(neutral_lms).to(device)
            neutral_lms = neutral_lms.repeat(num_mesh,1,1)

            path_n_mesh = "../S2D_Data_norm/neutral_mesh.mat"
            n_meshs = loadmat(path_n_mesh)
            n_meshs = n_meshs["neutral_mesh"].reshape(12,-1)
            n_meshs = np.array(n_meshs,dtype=np.float32)
            neutral_mesh = n_meshs[iden] # exp
            neutral_mesh = neutral_mesh.reshape(1, -1, 3) 
            neutral_mesh_ = np.zeros([1,5024,3])
            neutral_mesh_[:,:-1,:] = neutral_mesh
            neutral_mesh = torch.from_numpy(neutral_mesh_).to(device)
            neutral_mesh = neutral_mesh.repeat(num_mesh,1,1)
            neutral_mesh = neutral_mesh.to(dtype=torch.float32)
            
            
            if(len(input_lms.shape)==2):
                input_lms = input_lms.unsqueeze(0)

            dis_input = input_lms - neutral_lms


            output_dis_, output_points_, _ = model(dis_input,neutral_lms,neutral_mesh)

            output_points_ = output_points_.cpu().detach().numpy()

            output_points = output_points_[:,:-1,:]

            print(output_points)
            save_path_M = args_.save_path_Meshes+ '/level'+args_.level+'/exp_'+str(exp)+'_iden_'+str(iden)
            if not os.path.exists(save_path_M):
                os.makedirs(save_path_M)

            sample_diffs = np.diff(output_points, axis=0)
            print(sample_diffs)
            print(np.shape(output_points))
            np.save(os.path.join(save_path_M+'.npy'), output_points)

            save_meshes(output_points, save_path_M, n_meshes=num_mesh)

    