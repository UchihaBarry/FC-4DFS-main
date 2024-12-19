import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
from VAE_norm_ import VAE_model
import torch.utils.data as Data
from models.lm_lstm import *
# %%
def create_model(args, dataset,label_y,label_iden, evaluation=False, channels_list=None,):
    if args.last_gen_active == 'None':
        gen_last_active = None
    elif args.last_gen_active == 'Tanh':
        gen_last_active = nn.Tanh()
    else:
        raise Exception('Unrecognized last_gen_active')

    print("********** create num", "gen  *******************")
    gen = Generator_Model(args=args).to(args.device)
    

    
    return gen



def joint_train(args, data_X, data_y, data_iden, amps, gens, loss_recorder,lengths):
    """
    Train several stages jointly
    :param data_X, data_y, data_iden: Training samples(288,16,40), express label()
    :param gens: All previously trained stages     gens = gens[:curr_stage]
    :param gan_models: Models to be trained
    :param lengths: Lengths including current group
    :param amps: Amplitude for reconstruction nosie
    :param args: arguments
    :param loss_recorder: loss recorder
    """
    print("######################## start training ########################")
    loop = range(args.num_iters)
    if not args.silent:
        loop = tqdm(loop)
    device = args.device

    interpolator = partial(F.interpolate, mode='linear', align_corners=False)
    COMA = Data.TensorDataset(data_X, torch.Tensor(data_iden), torch.Tensor(data_y))
    Loader_COMA = Data.DataLoader(dataset=COMA,batch_size=args.size_batch,shuffle=args.enable_shuffle,num_workers = 8)
   
    
    optim = torch.optim.Adam(params = gens.parameters(), lr=args.lr_g)
    max_len = 40  
    min_len = 15 
    increment_step = 5 
    increment_iter = args.num_iters // 5 
    current_len = min_len 
    for epoch in tqdm(loop):
        
        gens.train()
    
        if epoch % increment_iter == 0:
            
            increment_times = epoch // increment_iter
            current_len = min_len + increment_times * increment_step
            current_len = min(current_len, max_len)  
        len = 30
        print("epoch:", epoch,'\t sequence length:', len)
        for step, (batch_real_0, batch_iden, batch_y) in enumerate(tqdm(Loader_COMA)):
            
            optim.zero_grad()
            
            batch_real_0_ = gt_interpolated(batch_real_0, len)
            gens.init_c_h(batch_real_0)
            
            data_lm_0 = batch_real_0_.permute(-1,0,1)
            
            batch_landmark = (data_lm_0.permute(1,2,0))
            
            pos = Pos_enconding(batch_landmark)
            

            label_y, iden_label = get_labels(y = batch_y ,iden_flag = batch_iden, args = args)
            
            iden_label = batch_landmark[:,:,0].squeeze(-1).to(args.device)
            '''forward and train'''
            length = batch_landmark.size()[-1]
            
            loss_func = CombinedLoss()
            
            generated_images_list = []

            for i in range(length):
                if i == 0:
                    input = iden_label
                    
                gt_ = batch_landmark[:,:,i].unsqueeze(-1).to(args.device)
                pos_input = pos[:,:,i].unsqueeze(-1).to(args.device).requires_grad_(False)
                
                generated_image, loss_expr = gens(label_y, iden_label, input, pos_input)
                input = generated_image
                generated_images_list.append(generated_image.squeeze(-1))

                
            batch_landmark = batch_landmark.to(args.device)
            generated_images = torch.stack(generated_images_list, dim=2).to(args.device)
            loss_cosine = 1 - F.cosine_similarity(generated_images, batch_landmark, dim=-1).mean()
            loss_temporal = torch.mean(torch.abs((generated_images[:,:,1:] - generated_images[:,:,:-1]) 
                                                 - (batch_landmark[:,:,1:] - batch_landmark[:,:,:-1])))
            loss_regular = loss_func(generated_images, batch_landmark)
            loss_L1 = loss_func.lossL1
            loss_L2 = loss_func.lossL2
            
            loss = loss_regular  + 0.25 * loss_temporal # + 0.001 * loss_cosine

            loss.backward(inputs = list(gens.parameters()),retain_graph=True)

            optim.step()
        loss_recorder.add_scalar('L1loss', loss_L1, epoch)
        loss_recorder.add_scalar('L2loss', loss_L2, epoch)
        print('loss:',loss,'\nlossL1:',loss_L1,'\nloss_temporal:',loss_temporal,'\nlossExpr:',loss_expr,'\nlosscosine:',loss_cosine)
        
        
           
                
        '''save, recoder loss'''
        
        steps_save_path = pjoin(args.save_path, 'models')
        if not os.path.exists(steps_save_path):
            os.makedirs(steps_save_path)
        if args.save_freq != 0 and (epoch + 1) % args.save_freq == 0:
            name = pjoin(steps_save_path, f'{(epoch + 1) // args.save_freq:03d}x.pt')
            
            torch.save(gens.state_dict(), name)
            
        
    print("################################\n\n\n\n\n\tfinishing joint training\n\n\n\n\n################################")


# %%
def loss_fn(gt, generated_image):
    L1_loss = nn.L1Loss()
    L2_loss = nn.MSELoss()
    loss = 0.1 * L1_loss(generated_image, gt) +  L2_loss(generated_image, gt)
    return loss 


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.L1_loss = nn.L1Loss()
        self.L2_loss = nn.MSELoss()
        self.lossL1 = 0
        self.lossL2 = 0

    def forward(self, generated_image, gt):
        loss = 0.1 * self.L1_loss(generated_image, gt) 
        
        self.lossL1 = 0.1 * self.L1_loss(generated_image, gt)
        self.lossL2 = self.L2_loss(generated_image, gt)
        return loss
