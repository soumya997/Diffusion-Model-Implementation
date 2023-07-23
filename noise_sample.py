
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm import Diffusion


class DDPM(nn.Module):
    def __init__(self, network, img_size, device, noise_steps=1000, beta_start=1e-4, beta_end=0.02):

        self.network = network()
        self.noise_steps=noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = torch.linspace(self.beta_start, self.beta_end, 
                                   self.noise_steps)
        self.alpha = 1 - self.beta
        
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, x, t):
        normal_noise = torch.randn_like(x)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1,1,1,1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).view(-1,1,1,1)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * normal_noise
        return x_t, normal_noise

    def reverse(self, x_t, t):
        return self.network(x_t, t)
    

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    def sample(self, network, n, in_channels):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, in_channels, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(self.noise_steps)):
                t = (torch.ones((n)) * i).long().to(self.device)
                print("t shape: ", t.shape)
                pred_noise = network(x, t)
                alpha = self.alpha[t].view(-1, 1, 1, 1)
                alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1, 1)
                if i>1:
                    eta = torch.randn_like(x)
                else:
                    eta = torch.zeros_like(x)
                
                x = (1/ torch.sqrt()) * (x - ((1 - alpha)/torch.sqrt(1 - alpha_hat)) * pred_noise) + torch.sqrt(beta) * eta
        
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x* 255).type(torch.uint8)
        return x



model = DDPM()
bs = 1
x = torch.randn(bs, 3, 64, 64)
timesteps = torch.randint(0, 1000, (bs,)).long()
diff = Diffusion()
diff.noise_images(x, timesteps)