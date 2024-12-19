
# %%
from torch.autograd import Variable
import os
import torch.nn.functional as F
import scipy.io
import numpy as np
import torch.nn as nn
from option import TrainOptionParser
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from utils import *
from scipy.io import loadmat
import time
from time import gmtime, strftime
import random

np.random.seed(2023)

    
def get_pyramid_lengths(args, dest):
    lengths = [16]
    ratio = eval(args.ratio)
    lengths[0] = int(dest * ratio)
    while lengths[-1] < dest:
        # scaling_rate default=1/0.75
        lengths.append(int(lengths[-1] * args.scaling_rate))
        if lengths[-1] == lengths[-2]:
            lengths[-1] += 1
    lengths[-1] = dest
    return lengths

def get_group_list(args, num_stages):
    group_list = []
    for i in range(0, num_stages, args.group_size):
        group_list.append(list(range(i, min(i + args.group_size, num_stages))))
    return group_list





def load_image(
        image_path,
        image_size=64,
        image_value_range=(-1, 1),
        is_gray=False,
        is_flip=False,
):
    if is_gray:
        image = imread(image_path, flatten=True).astype(np.float32)
    else:
        image = imread(image_path).astype(np.float32)
    if is_flip:
        image = flip(image)
    image = transform.resize(image, [image_size, image_size])
    image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
    return image


def save_batch_images(
        batch_images,
        save_path,
        image_value_range=(-1, 1),
        size_frame=None
):
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])
    for ind, image in enumerate(images):
        ind_col = ind % size_frame[1]
        ind_row = ind // size_frame[1]
        frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
    imsave(save_path, frame)


def save_single_images(
        batch_images,
        save_path,
        save_files,
        image_value_range=(-1, 1)
):
    images = (batch_images + image_value_range[1]) * 127.5
    for image, file_name in zip(images, save_files):
        image = np.squeeze(np.clip(image, 0, 255).astype(np.uint8))
        imsave(os.path.join(save_path, file_name), image)



def get_gpu_info():
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        return gpus[0].name if len(gpus) > 0 else 'NO GPU detected'
    except:
        return 'NO GPUtil installed'


def get_cpu_info():
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return info['brand_raw']
    except:
        return 'NO cpuinfo installed'


def get_device_info():
    return get_cpu_info(), get_gpu_info()
