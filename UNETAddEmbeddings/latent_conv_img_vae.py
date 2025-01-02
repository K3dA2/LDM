import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
import matplotlib.pyplot as plt
import uuid
import os
import cv2

class VAEConfig:
    def __init__(self, z_dim=16, width=64, in_channels=3, out_channels=32, device='cpu'):
        self.z_dim = z_dim
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

class ResNet(nn.Module):
    def __init__(self, in_channels = 3 ,out_channels = 32):
        super().__init__()
        num_groups = 4
        if in_channels < 4:
            num_groups = in_channels
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.network = nn.Sequential(
            nn.GroupNorm(num_groups,in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups,out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.network(x)
        return torch.add(out,self.residual_layer(x))

class Encoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(config.in_channels, config.width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(config.width, config.width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.width, config.width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(config.width * 2, config.width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            ResNet(config.width * 2, config.width * 4),
            nn.ReLU(),
            ResNet(config.width * 4, config.z_dim)
        )
        self.mu = nn.Conv2d(config.z_dim, config.z_dim, kernel_size=3, padding=1)
        self.sigma = nn.Conv2d(config.z_dim, config.z_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.net(x)
        mu = self.mu(x)
        l_sigma = self.sigma(x)
        z = mu + torch.exp(l_sigma / 2) * torch.rand_like(l_sigma)
        return mu, l_sigma, z

class Decoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            ResNet(config.z_dim, config.width * 4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            ResNet(config.width * 4, config.width * 4),
            nn.ReLU(),
            nn.Conv2d(config.width * 4, config.width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(config.width * 2, config.width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.width, config.width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.width, config.in_channels, kernel_size=3, padding=1)
        )

    def forward(self, z):
        z = self.net(z)
        return z

class DecoderGan(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            ResNet(config.z_dim, config.width),
            nn.ReLU(),
            nn.BatchNorm2d(config.width),
            nn.ConvTranspose2d(config.width, config.width * 4, kernel_size=2, stride=2),
            nn.ReLU(),
            ResNet(config.width * 4, config.width * 4),
            nn.ReLU(),
            nn.Conv2d(config.width * 4, config.width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(config.width * 2),
            nn.ConvTranspose2d(config.width * 2, config.width * 2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(config.width * 2, config.width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.width, config.width, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.width),
            nn.ReLU(),
            nn.Conv2d(config.width, config.in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.net(z)
        return z

class Discriminator(nn.Module):
    def __init__(self, width=32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(width * 2),
            nn.Conv2d(width * 2, width * 2, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(width * 4),
            nn.Conv2d(width * 4, width * 4, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(width * 4, width * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(width * 8),
            nn.Conv2d(width * 8, width * 8, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(width * 8, width * 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(width * 16),
            nn.Conv2d(width * 16, width * 16, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(width * 16, width, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(width),
            nn.LeakyReLU(),
            nn.Conv2d(width, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, img):
        emb = self.net(img)               # Pass the image through the convolutional layers
        return emb


class VariationalAutoencoderGAN(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = DecoderGan(config)
        self.device = config.device
        self.z_dim = config.z_dim
    
    def forward(self, x):
        mu, l_sigma, z = self.encoder.forward(x)
        out = self.decoder.forward(z)
        return mu, l_sigma, out
    
    def calculate_alpha(self, perceptual_loss, gan_loss):
        # Ensure the losses are scalars by taking the mean
        perceptual_loss_scalar = perceptual_loss.mean()
        gan_loss_scalar = gan_loss.mean()

        # Get the last layer's weights
        last_layer = self.decoder.net[-2]
        last_layer_weight = last_layer.weight

        # Calculate gradients for the scalar losses
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss_scalar, last_layer_weight, create_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss_scalar, last_layer_weight, create_graph=True)[0]

        # Calculate alpha
        alpha = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        return alpha.detach()  # Detach to stop gradient computation

    def decode(self,latents,should_save=True):
        self.decoder.eval()
        pred = self.decoder.forward(latents)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        pred = self.unnormalize(pred, mean, std)

        pred = np.transpose(pred[-1].cpu().detach().numpy(), (1, 2, 0))
        pred = np.clip(pred, 0, 1)
        
        if should_save:
            plt.imshow(pred)
            plt.axis('off')  # Hide the axes
            random_filename = str(uuid.uuid4()) + '.png'
            save_directory = 'UNETAddEmbeddings/Images'
            full_path = os.path.join(save_directory, random_filename)
            plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.imshow(np.transpose(pred[-1].cpu().detach().numpy(), (1, 2, 0)))
            plt.show()
        self.decoder.train()

    def inferenceR(self, should_save=True):
        z_var = torch.rand(1, self.z_dim, 16, 16).to(self.device)
        self.decoder.eval()
        pred = self.decoder.forward(z_var)
        if should_save:
            plt.imshow(np.transpose(pred[-1].cpu().detach().numpy(), (1, 2, 0)))
            plt.axis('off')  # Hide the axes
            random_filename = str(uuid.uuid4()) + '.png'
            save_directory = 'Images/'
            full_path = os.path.join(save_directory, random_filename)
            plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.imshow(np.transpose(pred[-1].cpu().detach().numpy(), (1, 2, 0)))
            plt.show()
        self.decoder.train()
    
    def unnormalize(self, tensor, mean, std):
        mean = torch.tensor(mean).to(tensor.device).view(1, -1, 1, 1)
        std = torch.tensor(std).to(tensor.device).view(1, -1, 1, 1)
        tensor = tensor * std + mean  # Out-of-place operations
        return tensor

    def reconstruct(self, image_tensor):
        self.decoder.eval()
        self.encoder.eval()

        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.ndim == 3:  # Single image, no batch dimension
                image_tensor = image_tensor.unsqueeze(0)
            elif image_tensor.ndim != 4:  # Expecting (B, C, H, W)
                raise ValueError("Expected image tensor to be in format (B, C, H, W)")

            image_tensor = image_tensor.float().to(self.device)
        else:
            raise TypeError("Expected input to be a torch.Tensor")

        _, _, l_img = self.encoder(image_tensor)
        img_rec = self.decoder(l_img)

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        img_rec = self.unnormalize(img_rec, mean, std)

        img_rec_np = np.transpose(img_rec[-1].cpu().detach().numpy(), (1, 2, 0))
        img_rec_np = np.clip(img_rec_np, 0, 1)

        plt.imshow(img_rec_np)
        plt.axis('off')

        random_filename = str(uuid.uuid4()) + '.png'
        save_directory = 'Reconstructed/'
        full_path = os.path.join(save_directory, random_filename)
        plt.savefig(full_path, bbox_inches='tight', pad_inches=0)

        self.decoder.train()
        self.encoder.train()


class TestEncoder(unittest.TestCase):
    def test_forward(self):
        config = VAEConfig(z_dim=1, width=64)
        model = VariationalAutoencoderGAN(config)
        input_tensor = torch.randn(1, 3, 64, 64)
        _, _, out = model.forward(input_tensor)
        self.assertEqual(out.shape, (1, 3, 64, 64))

if __name__ == "__main__":
    unittest.main()
