from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import os
import random


class SingleLoss:
    def __init__(self, name: str, writer: SummaryWriter, base=0):
        self.name = name
        self.loss_step = []
        self.loss_epoch = []
        self.loss_epoch_tmp = []
        self.writer = writer
        if base:
            self.loss_epoch = [0] * base
            self.loss_step = [0] * base * 10

    def add_event(self, val, step=None, name='scalar'):
        if step is None: step = len(self.loss_step)
        if val is None:
            val = 0
        else:
            callee = getattr(self.writer, 'add_' + name)
            callee(self.name + '_step', val, step)
        self.loss_step.append(val)
        self.loss_epoch_tmp.append(val)

    def epoch(self, step=None):
        if step is None: step = len(self.loss_epoch)
        loss_avg = sum(self.loss_epoch_tmp) / len(self.loss_epoch_tmp)
        self.loss_epoch_tmp = []
        self.loss_epoch.append(loss_avg)
        self.writer.add_scalar('Train/epoch_' + self.name, loss_avg, step)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        loss_step = np.array(self.loss_step)
        loss_epoch = np.array(self.loss_epoch)
        np.save(path + self.name + '_step.npy', loss_step)
        np.save(path + self.name + '_epoch.npy', loss_epoch)

    def last_epoch(self):
        return self.loss_epoch[-1]


class LossRecorder:
    def __init__(self, writer: SummaryWriter, base=0):
        self.losses = {}
        self.writer = writer
        self.base = base

    def add_scalar(self, name, val=None, step=None):
        if isinstance(val, torch.Tensor): val = val.item()
        if name not in self.losses:
            self.losses[name] = SingleLoss(name, self.writer, self.base)
        self.losses[name].add_event(val, step, 'scalar')

    def add_figure(self, name, val, step=None):
        if name not in self.losses:
            self.losses[name] = SingleLoss(name, self.writer, self.base)
        self.losses[name].add_event(val, step, 'figure')

    def verbose(self):
        lst = {}
        for key in self.losses.keys():
            lst[key] = self.losses[key].loss_step[-1]
        lst = sorted(lst.items(), key=lambda x: x[0])
        return str(lst)

    def epoch(self, step=None):
        for loss in self.losses.values():
            loss.epoch(step)

    def save(self, path):
        for loss in self.losses.values():
            loss.save(path)

    def last_epoch(self):
        res = []
        for loss in self.losses.values():
            res.append(loss.last_epoch())
        return res

class GAN_loss(nn.Module):
    def __init__(self, gan_mode, real_label=1.0, fake_label=0.0):
        super(GAN_loss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgan-gp':
            self.loss = self.wgan_loss
            real_label = 1
            fake_label = 0
        elif gan_mode == 'none':
            self.loss = None
        else:
            raise Exception('Unknown GAN mode')

        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

    @staticmethod
    def wgan_loss(prediction, target):
        lmbda = torch.ones_like(target)
        lmbda[target == 1] = -1
        return (prediction * lmbda).mean()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class RecLoss(nn.Module):
    def __init__(self, real_data, fake_data=False, loss_type='L', ):
        super(RecLoss, self).__init__()
        self.real_data = real_data
        self.fake_data = fake_data
        # self.lambda_pos = lambda_pos
        # self.extra_velo = extra_velo
        if loss_type == 'L2':
            self.criteria = nn.MSELoss()
        elif loss_type == 'L1':
            self.criteria = nn.L1Loss()
        elif loss_type == 'L':
            self.criteria = nn.L1Loss() + nn.MSELoss()
        else:
            raise Exception('Unknown loss type')

    def __call__(self, a, b):
        # if self.lambda_pos > 0:
        a_pos = self.real_data
        b_pos = self.fake_data
        # loss_pos = self.criteria(a_pos, b_pos)
        
        return self.criteria(a, b) 

class ImagePool:
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

