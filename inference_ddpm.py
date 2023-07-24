import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from ddpm_unet import unet_conv
from ddpm import DDPM
import sys
import logging
import numpy as np
from config import config
import random
import cv2
from PIL import Image

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


# def load_model(path):
#     model = Net().cuda()
#     model.load_state_dict(torch.load(path, map_location=DEVICE)['state_dict'])
#     model.eval()
#     return model

BASE = "/home/alive/somusan/diffusion_model/scratch-diffusion-model"
device = "cuda"
model_path = "/home/alive/somusan/diffusion_model/scratch-diffusion-model/models/DDPM_CONVOLUTIONAL/86_ckpt.pt"
model = unet_conv(device=device).to(device)
weights = torch.load(model_path, map_location=device)
# print(weights.keys())
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
with torch.no_grad():
    diffusion = DDPM(img_size=config.image_size, device=device)
    sampled_images = diffusion.sample(model, n=4, in_channels=3)
    print(sampled_images.shape)
    save_images(sampled_images, os.path.join(BASE, "inference", f"{69}.jpg"))

    print('\n')
    for i in range(sampled_images.shape[0]):
        img = sampled_images[i,:,:,:]
        img = img.to('cpu').permute(1,2,0).numpy()
        img = cv2.resize(img, (256,256))
        print(img.shape)
        im = Image.fromarray(img)
        im.save(os.path.join(BASE, "inference", f"{i}_infer.jpg"))
        # print(img.shape)
        # cv2.imwrite(os.path.join(BASE, "inference", f"{i}_infer.jpg"), img)
