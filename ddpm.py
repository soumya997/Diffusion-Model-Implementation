
import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpm_unet import unet_conv
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_data
from torchvision import transforms 
import argparse
import torchvision
from torchvision.utils import save_image




class DDPM(nn.Module):
    def __init__(self, img_size, device, is_cfg = False, noise_steps=1000, beta_start=1e-4, beta_end=0.02):

        self.noise_steps=noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = torch.linspace(self.beta_start, self.beta_end, 
                                   self.noise_steps).to(self.device)
        self.alpha = 1 - self.beta
        
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, x, t):
        normal_noise = torch.randn_like(x)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1,1,1,1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).view(-1,1,1,1)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * normal_noise
        return x_t, normal_noise

    def reverse(self, network, x_t, t):
        return network(x_t, t)
    
    def reverse_cfg(self, network, x_t, t, y):
        return network(x_t, t, y)
    

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    def sample(self, network, n, in_channels):

        network.eval()
        with torch.no_grad():
            x = torch.randn((n, in_channels, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(self.noise_steps)):
                t = (torch.ones((n)) * i).long().to(self.device)
                # print("t shape: ", t.shape)
                pred_noise = network(x, t)
                alpha = self.alpha[t].view(-1, 1, 1, 1)
                alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1, 1)
                if i>1:
                    eta = torch.randn_like(x)
                else:
                    eta = torch.zeros_like(x)
                
                x = (1/ torch.sqrt(alpha)) * (x - ((1 - alpha)/torch.sqrt((1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * eta
        
        network.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x* 255).type(torch.uint8)
        return x
    

    def sample_cfg(self, model, n, labels, cfg_scale=3):
        
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t].view(-1, 1, 1, 1)
                alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1, 1)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 16  # 5
    args.image_size = 64
    args.dataset_path = r"/home/alive/somusan/diffusion_model/Diffusion-Models-pytorch/pictures"

    dataloader = get_data(args)

    # diff = Diffusion(device="cpu")
    diff = DDPM(img_size=64, device="cpu")

    image = next(iter(dataloader))[0][10, :,:,:]

    t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long()
    noised_image, _ = diff.add_noise(image, t)


    new_x_np = noised_image.detach().cpu().permute(3,2,1,0).squeeze(dim=3).numpy()

    save_image(noised_image.add(1).mul(0.5), "noise.jpg")

    plt.imshow(new_x_np)
    plt.show()

        