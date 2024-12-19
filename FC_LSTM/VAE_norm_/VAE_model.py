# %%
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torch
        
        

# %%
latent_dim = 16
input_dim = 68 * 3
inter_dim = [128,64,32]
device = 0

class AE(nn.Module):
    def __init__(self, input_dim=input_dim, inter_dim=inter_dim, latent_dim=latent_dim, device=device):
        super(AE, self).__init__()
        self.batch = 1
        self.org_size = [1,204]
        
       
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, inter_dim[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(inter_dim[0], inter_dim[1], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(inter_dim[1], inter_dim[2], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(inter_dim[2], latent_dim, kernel_size=1, stride=1, padding=0),
        ).to(device)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, inter_dim[2], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(inter_dim[2], inter_dim[1], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(inter_dim[1], inter_dim[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(inter_dim[0], input_dim, kernel_size=3, stride=1, padding=1),
        ).to(device)


    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        self.org_size = x.size()
        self.batch = self.org_size[0]
        
        x = x.view(self.batch, -1, 1)

        z = self.encoder(x)
        
        recon_x = self.decoder(z).view(size=self.org_size)

        return recon_x
    
    def forward_en(self,x):
        self.org_size = x.size()
        self.batch = self.org_size[0]
        
        z = self.encoder(x)
        
        return z
    
    def forward_de(self,z):
        
        recon_x = self.decoder(z).view(size=self.org_size)

        return recon_x

class AE_en(nn.Module):
    def __init__(self, input_dim=input_dim, inter_dim=inter_dim, latent_dim=latent_dim, device=device):
        super(AE_en, self).__init__()
        self.batch = 1
        self.org_size = [1,204]
        self.device = device
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, inter_dim[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(inter_dim[0], inter_dim[1], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(inter_dim[1], inter_dim[2], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(inter_dim[2], latent_dim, kernel_size=1, stride=1, padding=0),
        ).to(device)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, inter_dim[2], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(inter_dim[2], inter_dim[1], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(inter_dim[1], inter_dim[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(inter_dim[0], input_dim, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid(),
        ).to(device)


    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)
    
    def forward(self,x):
        self.org_size = x.size()
        self.batch = self.org_size[0]
    
        x = x.view(self.batch, -1,1).to(self.device)

        z = self.encoder(x).squeeze()
      
        return z

class AE_de(nn.Module):
    def __init__(self, input_dim=input_dim, inter_dim=inter_dim, latent_dim=latent_dim, device=device):
        super(AE_de, self).__init__()
        self.batch = 1
        self.org_size = [1,204]
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, inter_dim[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(inter_dim[0], inter_dim[1], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(inter_dim[1], inter_dim[2], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(inter_dim[2], latent_dim, kernel_size=1, stride=1, padding=0),
        ).to(device)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, inter_dim[2], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(inter_dim[2], inter_dim[1], kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(inter_dim[1], inter_dim[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(inter_dim[0], input_dim, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid(),
        ).to(device)

        
    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    
    
    def forward(self,z):
        self.batch = z.size()[0]
        self.org_size[0] = self.batch
        z = z.unsqueeze(-1)
        recon_x = self.decoder(z).view(size=self.org_size)

        return recon_x

