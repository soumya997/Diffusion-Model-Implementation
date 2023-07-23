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


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

BASE = "/home/alive/somusan/diffusion_model/scratch-diffusion-model"
min_loss = sys.maxsize
loss_profile = []

def train(config):
    global min_loss, loss_profile
    setup_logging(config.run_name)
    device = config.device
    dataloader = get_data(config)
    model = unet_conv(device="cuda").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    diffusion = DDPM(img_size=config.image_size, device=device)


    for epoch in range(config.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.add_noise(images, t)
            predicted_noise = diffusion.reverse(network=model, x_t=x_t, t=t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(EPOCH= epoch+1,MSE=loss.item())

        sampled_images = diffusion.sample(model, n=images.shape[0]+1, in_channels=images.shape[1])
        save_images(sampled_images, os.path.join(BASE, "results", config.run_name, f"{epoch}.jpg"))
        loss_profile.append(loss.item())
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), os.path.join(BASE, "models", config.run_name, f"{epoch}_ckpt.pt"))


def launch():
    conf = config()
    train(conf)
    loss_file_name = os.path.join(BASE, "logs", 
                                    conf.run_name + "_" + str(conf.run_num) + "_loss.npy")
    np.save(loss_file_name, np.array(loss_profile))


if __name__ == '__main__':
    launch()
