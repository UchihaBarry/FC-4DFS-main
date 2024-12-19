"""
    need to change:
    data_path(nerual root)
    train data()
"""

import numpy as np
import json
import os
import copy
import _pickle as pickle
import torch.nn.functional as F
# import mesh_sampling
import trimesh

# from get_landmarks import get_landmarks
from autoencoder_dataset import autoencoder_dataset
from torch.utils.data import DataLoader
# from spiral_utils import get_adj_trigs, generate_spirals
from models import SpiralAutoencoder
# from test_funcs import test_autoencoder_dataloader
import torch
from train_funcs import train_autoencoder_dataloader
from tensorboardX import SummaryWriter
from sklearn.metrics.pairwise import euclidean_distances
# from save_meshes import save_meshes
from PIL import Image
import glob
import argparse
import VAE_model
####################


parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('--test_fold', type=int, help='Change the fold here from 1 to 4', default=1)
parser.add_argument('--Split', type=str, help='Expressions split protocol: Split=Expr, Id split: Split=Id', default='Id')
parser.add_argument('--Lands', type=str, help='Lands=GT if you want to reproduce table 1 results, Lands=Motion3DGAN if you want to show Motion3DGAN generated samples', default='GT')
parser.add_argument('--label', type=str, help='the desired label for Motion3DGAN samples', default='0')
args = parser.parse_args()

Split=args.Split
fold=args.test_fold
Lands=args.Lands

#####################################

root_dir = '../S2D_Data_norm'
results_dir = './Models/train'
testresults_dir = './Results/tst'


template_dir = './template/'
meshpackage = 'trimesh'
GPU = True
device_idx = 0
torch.cuda.get_device_name(device_idx)
#########################

args = {}
downsample_method = 'COMA_downsample'

reference_mesh_file = os.path.join(template_dir, 'template', 'template.obj')
downsample_directory = os.path.join(template_dir, 'template', downsample_method)
ds_factors = [4, 4, 4, 4]
step_sizes = [2, 2, 1, 1, 1]
filter_sizes_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
dilation_flag = True
if dilation_flag:
    dilation = [2, 2, 1, 1, 1]
else:
    dilation = None
reference_points = [[3567, 4051, 4597]]# [[414]]
name = ''
dataset = 'CoMA'
args = { 'neutral_data': os.path.join(root_dir),
        'mode':'train','generative_model':'autoencoder',
        'name': name, 
        # 'data': os.path.join(root_dir, dataset, 'preprocessed',name),
        'results_folder': os.path.join(results_dir),
        'testresults_folder': os.path.join(testresults_dir),
        'save_path_Meshes': os.path.join(testresults_dir,'predicted_meshes'),
        'save_path_animations': './Results/Gifs/',
        'reference_mesh_file': reference_mesh_file, 'downsample_directory': downsample_directory,
        'checkpoint_file': 'checkpoint',
        'seed': 2, 'loss': 'l1',
        'batch_size': 64, 'num_epochs': 800, 'eval_frequency': 200, 'num_workers': 4,
        'filter_sizes_enc': filter_sizes_enc, 'filter_sizes_dec': filter_sizes_dec,
        'nz': 16,
        'ds_factors': ds_factors, 'step_sizes': step_sizes, 'dilation': dilation,
        'lr': 1e-5,
        'regularization': 5e-5,
        'scheduler': True, 'decay_rate': 0.99, 'decay_steps': 1,
        'resume': False, 'nbr_landmarks': 68,
        'shuffle': True, 'nVal': 100, 'normalization': False}

args['results_folder'] = os.path.join(args['results_folder'],'latent_'+str(args['nz']))


if not os.path.exists(os.path.join(args['results_folder'])):
    os.makedirs(os.path.join(args['results_folder']))

checkpoint_path = os.path.join(args['results_folder'],'checkpoints', args['name'])
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

prediction_path = os.path.join(args['testresults_folder'], 'predictions')
if not os.path.exists(prediction_path):
    os.makedirs(prediction_path)


samples_path = os.path.join(args['results_folder'],'samples', args['name'])
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

summary_path = os.path.join(args['results_folder'],'summaries',args['name'])
if not os.path.exists(summary_path):
    os.makedirs(summary_path)  

if not os.path.exists(args['save_path_Meshes']):
    os.makedirs(args['save_path_Meshes'])

if not os.path.exists(downsample_directory):
    os.makedirs(downsample_directory)

#####################################################
if __name__ == "__main__":
    np.random.seed(args['seed'])
    print("Loading data .. ")
    
    

    # ###################################################

    torch.manual_seed(args['seed'])

    if GPU:
        device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    #################################################

    dataset_train = autoencoder_dataset(neutral_root_dir=args['neutral_data'], points_dataset='train',
                                       normalization = args['normalization'], template = reference_mesh_file)

    dataset_val = autoencoder_dataset(neutral_root_dir=args['neutral_data'], points_dataset='val',
                                       normalization = args['normalization'], template = reference_mesh_file)
    dataloader_train = DataLoader(dataset_train, batch_size=args['batch_size'],
                                 shuffle=False, num_workers=args['num_workers'])
    dataloader_val = DataLoader(dataset_val, batch_size=args['batch_size'],
                                 shuffle=False, num_workers=args['num_workers'])

    model = VAE_model.AE(input_dim=68*3,inter_dim=[128,64,32],latent_dim=16,device = device)


    

    optim = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['regularization'])
    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, args['decay_steps'], gamma=args['decay_rate'])
    else:
        scheduler = None

    if args['loss'] == 'l1':
        def loss_l1(outputs, targets, points_dis, displacement):
            weights = np.load(template_dir + './template/Normalized_d_weights.npy')
            Weigths = torch.from_numpy(weights).float().to(device)
            
            # target_expression = outputs - inputs # target_displacement
            L = (torch.matmul(Weigths, torch.abs(outputs - targets))).mean() +  0.1 * torch.abs(
                points_dis - displacement).mean()
            # L = torch.abs(points_dis - displacement).mean()
            return L
        
        kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = lambda recon_x, x: F.mse_loss(recon_x, x, size_average=False)
       

        loss_fn = loss_l1
    #########################################################
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params)) 
    print(model)

    if args['mode'] == 'train':
        writer = SummaryWriter(summary_path)
        with open(os.path.join(args['results_folder'],'checkpoints', args['name'] +'_params.json'),'w') as fp:
            saveparams = copy.deepcopy(args)
            json.dump(saveparams, fp)
            
        if args['resume']:
                print('loading checkpoint from file %s'%(os.path.join(checkpoint_path,args['checkpoint_file'])))
                checkpoint_dict = torch.load(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar'),map_location=device)
                start_epoch = checkpoint_dict['epoch'] + 1
                model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
                optim.load_state_dict(checkpoint_dict['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
                print('Resuming from epoch %s'%(str(start_epoch)))     
        else:
            start_epoch = 0
            
        if args['generative_model'] == 'autoencoder':
            train_autoencoder_dataloader(dataloader_train, dataloader_val,
                            device, model, optim, loss_fn,kl_loss,recon_loss,
                            bsize = args['batch_size'],
                            start_epoch = start_epoch,
                            n_epochs = args['num_epochs'],
                            eval_freq = args['eval_frequency'],
                            scheduler = scheduler,
                            writer = writer,
                            metadata_dir=checkpoint_path, samples_dir=samples_path,
                            checkpoint_path = args['checkpoint_file'])
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n\t\tfinishing training\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")