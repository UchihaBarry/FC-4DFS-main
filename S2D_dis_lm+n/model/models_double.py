## Code modified from https://github.com/gbouritsas/Neural3DMM

import torch
import torch.nn as nn

import pdb

class SpiralConv(nn.Module):
    def __init__(self, in_c, spiral_size,out_c,activation='elu',bias=True,device=None):
        super(SpiralConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.device = device

        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self,x,spiral_adj):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()
  
        spirals_index = spiral_adj.view(bsize*num_pts*spiral_size) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=self.device).view(-1,1).repeat([1,num_pts*spiral_size]).view(-1).long() # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index,spirals_index,:].view(bsize*num_pts,spiral_size*feats) # [bsize*numpt, spiral*feats]


        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize,num_pts,self.out_c)
        zero_padding = torch.ones((1,x.size(1),1), device=self.device)
        zero_padding[0,-1,0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat

class SpiralAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, nbr_landmarks, sizes, spiral_sizes, spirals, D, U, device, activation = 'elu'):
        super(SpiralAutoencoder,self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spirals = spirals
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes
        self.D = [ d.to(device) for d in D ]
        self.U = [ u.to(device) for u in U ]
        self.device = device
        self.activation = activation
        self.nbr_landmarks = nbr_landmarks
        
        self.lambda_atten = nn.Parameter(torch.tensor(0.1)).to(device)
        self.para_loss = nn.Parameter(torch.tensor(300.)).to(device)
        self.para = nn.Parameter(0.1*torch.ones(len(spiral_sizes)).to(device))
        self.para_atten = nn.Parameter(0.1*torch.ones(len(spiral_sizes)).to(device))
        
        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes)-1):
            if filters_enc[1][i]:
                self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[1][i],
                                            activation=self.activation, device=device).to(device))
                input_size = filters_enc[1][i]

            self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[0][i+1],
                                        activation=self.activation, device=device).to(device))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv).to(device) 
        

        ### Check heeeeeeere
        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size).to(device)
        self.fc_latent_dec = nn.Linear(nbr_landmarks*3, (sizes[-1]+1)*filters_dec[0][0]).to(device)
        self.fc_latent_dec1 = nn.Linear(nbr_landmarks*3, (sizes[-1]+1)*filters_dec[0][0]).to(device)
        
        self.dconv_1 = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.dconv_1.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                             activation=self.activation, device=device).to(device))
                input_size = filters_dec[0][i+1]  
                
                if filters_dec[1][i+1]:
                    self.dconv_1.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    self.dconv_1.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[0][i+1]                      
                    self.dconv_1.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation='identity', device=device).to(device)) 
                    input_size = filters_dec[1][i+1] 
                else:
                    self.dconv_1.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[0][i+1]                      
                    
        self.dconv_1 = nn.ModuleList(self.dconv_1).to(device)
        
        self.dconv_2 = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.dconv_2.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                             activation=self.activation, device=device).to(device))
                input_size = filters_dec[0][i+1]  
                
                if filters_dec[1][i+1]:
                    self.dconv_2.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    self.dconv_2.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[0][i+1]                      
                    self.dconv_2.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation='identity', device=device).to(device)) 
                    input_size = filters_dec[1][i+1] 
                else:
                    self.dconv_2.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[0][i+1]                      
                    
        self.dconv_2 = nn.ModuleList(self.dconv_2).to(device)
        
        self.atten = []
        for i in range(len(spiral_sizes)-2):
            self.atten.append(nn.MultiheadAttention(embed_dim=filters_dec[0][i], num_heads=4,dropout=0.2,batch_first=True))
        self.atten = nn.ModuleList(self.atten).to(device)

    def encode(self,x):
        bsize = x.size(0)
        S = self.spirals
        D = self.D
        X=[]
        
        j = 0
        for i in range(len(self.spiral_sizes)-1):
            x = self.conv[j](x,S[i].repeat(bsize,1,1))
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,S[i].repeat(bsize,1,1))
                j+=1
            x = torch.matmul(D[i],x)
            X.append(x)
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x), X
    
    def cos(self,x_1,x_2):
        # 计算每个点的余弦相似度
        x_1_norm = nn.functional.normalize(x_1, p=2, dim=2)  # 沿最后一个维度归一化
        x_2_norm = nn.functional.normalize(x_2, p=2, dim=2)
        cos_sim_per_point = torch.sum(x_1_norm * x_2_norm, dim=2)

        # 对所有点的余弦相似度求平均，得到每个样本的平均相似度
        cos_sim = cos_sim_per_point.mean(dim=1)
        return 1 - cos_sim.mean()
  
    
    def decoder(self, z, z_n, X):
        bsize = z.size(0)
        S = self.spirals
        U = self.U
        x = self.fc_latent_dec(z)
        x_n = self.fc_latent_dec1(z_n)
        x_1 = x + x_n * self.lambda_atten
        x_1 = x_1.view(bsize,self.sizes[-1]+1,-1)
        x_2 = x.view(bsize,self.sizes[-1]+1,-1)
        j=0
        for i in range(len(self.spiral_sizes)-1):
            x_1 = torch.matmul(U[-1-i],x_1)
            x_1 = self.dconv_1[j](x_1,S[-2-i].repeat(bsize,1,1))
            if i<3:
                x_2 = x_2 + self.para[i] * X[-i-1]
            x_2 = torch.matmul(U[-1-i],x_2)
            x_2 = self.dconv_2[j](x_2,S[-2-i].repeat(bsize,1,1))
            # 1_atten_2
            if i<2:
                atten ,_ = self.atten[i+1](x_1,x_2,x_2)
                x_1 = x_1 + self.para_atten[i+1] * atten
            j+=1
            if self.filters_dec[1][i+1]: 
                x_1 = self.dconv_1[j](x_1,S[-2-i].repeat(bsize,1,1))
                x_2 = self.dconv_2[j](x_2,S[-2-i].repeat(bsize,1,1))
                j+=1
        # align_loss = nn.MSELoss()
        align_loss = nn.L1Loss()

        feature_align_loss = align_loss(x_1,x_2)*self.para_loss
        return x_2, feature_align_loss

        
        
    
    def forward(self, landmarks_dis, landmark_n, neutral_points):

        landmarks_dis = landmarks_dis.reshape(landmarks_dis.size()[0], -1)
        landmark_n = landmark_n.reshape(landmarks_dis.size()[0], -1)
        
        _, x = self.encode(neutral_points)
        X, loss_align = self.decoder(landmarks_dis, landmark_n, x) # displacement -> landmarks(already done)
        X_=X + neutral_points # from displacement to points
        return X, X_ ,loss_align 

