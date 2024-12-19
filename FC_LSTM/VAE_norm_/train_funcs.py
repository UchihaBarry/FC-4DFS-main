import os
import torch
from tqdm import tqdm
import numpy as np

def train_autoencoder_dataloader(dataloader_train, dataloader_val,
                                 device, model, optim, loss_fn, kl_loss,recon_loss,
                                 bsize, start_epoch, n_epochs, eval_freq, scheduler = None,
                                 writer=None,
                                 metadata_dir=None, samples_dir = None, checkpoint_path = None):
    
    total_steps = start_epoch*len(dataloader_train)

    for epoch in range(start_epoch, n_epochs):
        model.train()

        tloss = []
        for b, sample_dict in enumerate(tqdm(dataloader_train)):
            optim.zero_grad()
                
            
            landmarks = sample_dict['landmarks'].to(device)
            
            landmarks=torch.squeeze(landmarks)
            
            # 64,38,3
            cur_bsize = landmarks.shape[0]
            # landmarks = landmarks.reshape(cur_bsize,-1)
            
            recon_landmarks  = model(landmarks)
            # loss_kl = kl_loss(mu, logvar)
            loss_recon = recon_loss(recon_landmarks,landmarks)
            
            loss =  loss_recon
            
            loss = loss/cur_bsize

            loss.backward()
            optim.step()
            
            
            with torch.no_grad():    
                tloss.append(cur_bsize * loss.item())
            # loss = loss / cur_bsize

            if writer and total_steps % eval_freq == 0:
                writer.add_scalar('loss/loss/data_loss',loss.item(),total_steps)
                writer.add_scalar('training/learning_rate', optim.param_groups[0]['lr'],total_steps)
            total_steps += 1

        # validate
        model.eval()
        vloss = []
        with torch.no_grad():
            for b, sample_dict in enumerate(tqdm(dataloader_val)):

                andmarks = sample_dict['landmarks'].to(device)
            
                landmarks=torch.squeeze(landmarks)
                
                # 64,38,3
                cur_bsize = landmarks.shape[0]
                # landmarks = landmarks.reshape(cur_bsize,-1)
                
                recon_landmarks = model(landmarks)
                # loss_kl = kl_loss(mu, logvar)
                loss_recon = recon_loss(recon_landmarks,landmarks)
                
                loss = loss_recon
                
                loss = loss/cur_bsize

                # loss.backward()
                
                with torch.no_grad():
                    vloss.append(cur_bsize * loss.item())
                

        if scheduler:
            scheduler.step()
            
        epoch_tloss = sum(tloss) / float(len(dataloader_train.dataset))
        writer.add_scalar('avg_epoch_train_loss',epoch_tloss,epoch)
        if len(dataloader_val.dataset) > 0:
            epoch_vloss = sum(vloss) / float(len(dataloader_val.dataset))
            writer.add_scalar('avg_epoch_valid_loss', epoch_vloss,epoch)
            print('epoch {0} | tr {1} | val {2}'.format(epoch,epoch_tloss,epoch_vloss))
        else:
            print('epoch {0} | tr {1} '.format(epoch,epoch_tloss))
        model = model.cpu()
  
        torch.save({'epoch': epoch,
            'autoencoder_state_dict': model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        },os.path.join(metadata_dir, checkpoint_path+'.pth.tar'))
        
        if epoch % 50 == 0:
            torch.save({'epoch': epoch,
            'autoencoder_state_dict': model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            },os.path.join(metadata_dir, checkpoint_path+'%s.pth.tar'%(epoch)))

        model = model.to(device)

        

    print('~FIN~')


