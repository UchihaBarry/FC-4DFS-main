# from models.lm_lstm import *
from option import *
from utils import *
from loss import *
from architecture import *
from os.path import join as pjoin
from scipy.io import loadmat,savemat
from torch.autograd import Variable


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
    add code below to train.py
    parser.save(pjoin(args.save_path, 'args.txt'))
    os.makedirs(args.save_path, exist_ok=True)
"""
def process_landmarks(landmarks):
    points = np.zeros(np.shape(landmarks))
    mu_x = np.mean(landmarks[:, 0])
    mu_y = np.mean(landmarks[:, 1])
    mu_z = np.mean(landmarks[:, 2])
    mu = [mu_x, mu_y, mu_z]

    landmarks_gram=np.zeros(np.shape(landmarks))
    for j in range(np.shape(landmarks)[0]):
        landmarks_gram[j,:]= np.squeeze(landmarks[j,:])-np.transpose(mu)

    normFro = np.sqrt(np.trace(np.matmul(landmarks_gram, np.transpose(landmarks_gram))))
    land = landmarks_gram / normFro
    points[:,:]=land
    return points


def load_gens(args_train, save_path, device, data_path = '../motion_Data_norm/COMA_landmarks_neutral2Exp/'):
    train_parser = TrainOptionParser()
    args = train_parser.load(pjoin(save_path, 'args.txt'))
    args.device = device
    args.save_path = save_path
    device = torch.device(args.device)

    files, data_y, data_iden = load_data(args=args_train, path_=data_path)

    data_X = [read_landmark(path_landmark=data_path + file_lm)for file_lm in files] # list

    
    data_X = frame_interpolate(data_X)

    lengths = get_pyramid_lengths(args_train,args_train.dest)
    args_train.size_SRVF_W = lengths


    
    
    create = create_model
    args_train.size_batch = 1
    gen  = create(args_train, data_X, data_y, data_iden, evaluation=True)
    try:
        gen_sate = torch.load(pjoin(args.save_path, 'sota.pt'), map_location=args.device)
    except FileNotFoundError:
        pass
    gen.load_state_dict(gen_sate)
    
    amps = torch.load(pjoin(args.save_path, str(args.exp), 'amps.pt'), map_location=args.device)

    return gen, amps, args, lengths

def gen_noise(n_channel, length, full_noise, device):
    if full_noise:
        res = torch.randn((1, n_channel, length)).to(device)
    else:
        res = torch.randn((1, 1, length)).repeat(1, n_channel, 1).to(device)
    return res


def main():
    parser = TrainOptionParser()
    train_args = parser.parse_args()
    test_parser = TestOptionParser()
    test_args = test_parser.parse_args()


    path_iden_label = "../motion_Data_norm/iden_landmarks.mat"
    iden_labels = loadmat(path_iden_label)
    iden_labels = iden_labels["iden_landmarks"].reshape(12,-1)
    iden_labels = np.array(iden_labels,dtype=np.float32)

    gens, amps, args, lengths = load_gens(train_args, test_args.save_path, test_args.device, )
    args.size_batch = 1
    num_samples = 2


    z_star = torch.randn((1, 204, 1), device=args.device)
    gens.eval()
    for exp in range(12):
        for iden in range(12):

                lm_label_ = iden
                exp_label_ = exp
                lm_label = iden_labels[lm_label_]

                lm_label = torch.from_numpy(lm_label).to(args.device)
                lm_label = lm_label.unsqueeze(0)

                exp_label = y_to_rb_label(exp_label_,args)
                exp_label = exp_label.astype(np.float32)
                exp_label = torch.from_numpy(exp_label).to(args.device)
                exp_label = exp_label.unsqueeze(0) # (1, 204)
                

                lm_label = lm_label.unsqueeze(0)

                length = 30
                pos_ = torch.zeros([1,204,length])
                pos = Pos_enconding(pos_)
                imgs = torch.zeros([1,204,length])
                input = lm_label.unsqueeze(-1)
                gens.init_c_h(pos)
                generated_images_list = []
                for frame in range(length):
                    pos_input = pos[:,:,frame].unsqueeze(-1).to(args.device).requires_grad_(False)
                    generated_image, loss_expr = gens(exp_label, lm_label, input, pos_input,)
                    input = generated_image
                    generated_images_list.append(generated_image.squeeze(-1))
                generated_images = torch.stack(generated_images_list, dim=2).to(args.device).detach()
            
                img = generated_images.permute(1,2,0)
                sample = img.reshape(1,68,3,-1)

                sample = sample.permute(0,3,1,2)
                sample = sample.squeeze()


                

                img = sample.cpu().detach().numpy()

                save_path = './generate_sample_pos_1_lstm/level'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path = save_path + "/exp_"+str(exp_label_)+"_iden_"+str(lm_label_)

                savemat(os.path.join(save_path+".mat") , {"img": sample})


                img = sample.cpu().detach().numpy()
                sample_diffs = np.diff(img, axis=0)
                print(sample_diffs)
                np.save(os.path.join(save_path+'.npy'), img)
if __name__ == '__main__':
    main()