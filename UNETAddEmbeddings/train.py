import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset,Dataset
from PIL import Image
from tqdm import tqdm
import datetime
import os
import torch.nn.utils as utils
from model import UnetClipConditional,Config,Unet
from utils import forward_cosine_noise, reverse_diffusion_cfg, count_parameters,reverse_diffusion
import random
import matplotlib.pyplot as plt
from latent_conv_img_vae import VariationalAutoencoderGAN, VAEConfig
import pandas as pd

class CelebADataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, subset_ratio=1.0):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            subset_ratio (float): Fraction of the dataset to use (0 < subset_ratio <= 1.0).
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Restrict the dataset to the specified subset
        if subset_ratio < 1.0:
            subset_size = int(len(self.data) * subset_ratio)
            self.data = self.data.sample(n=subset_size, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the image file name and attributes
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        attributes = self.data.iloc[idx, 1:].values.astype(int)

        # Convert -1 to 0 for binary tensor
        attributes = torch.tensor((attributes == 1).astype(int), dtype=torch.float32)

        # Open and preprocess the image
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, attributes

def show_images_and_noise(images, noise, num_images=5):
    images = images[:num_images].cpu().detach().numpy().transpose(0, 2, 3, 1)
    noise = noise[:num_images].cpu().detach().numpy().transpose(0, 2, 3, 1)
    
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    
    for i in range(num_images):
        axes[0, i].imshow((images[i] * 0.5) + 0.5)  # Denormalize to [0, 1]
        axes[0, i].axis('off')
        axes[0, i].set_title("Original Image")
        
        axes[1, i].imshow((noise[i] * 0.5) + 0.5)  # Denormalize to [0, 1]
        axes[1, i].axis('off')
        axes[1, i].set_title("Noise")

    plt.show()

def normalize_latents(latent, mean, std):
    return (latent - mean) / std

def unscale_latents(latent_normalized, mean, std):
    return latent_normalized * std + mean

def training_loop(n_epochs, optimizer, model, loss_fn, device, data_loader, vae ,max_grad_norm=1.0, timesteps=1000, epoch_start=0, accumulation_steps=4):
    for epoch in range(epoch_start, n_epochs + epoch_start):
        model.train()
        loss_train = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit=' batch')
        optimizer.zero_grad()  # Initialize the gradient

        for batch_idx, (imgs, labels) in enumerate(progress_bar):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Generate timestamps
            t = torch.randint(0, timesteps, (imgs.size(0),), dtype=torch.float32).to(device) / timesteps
            t = t.view(-1, 1)

            #Turn Images to latents
            with torch.no_grad():
                _,_,imgs = vae.encoder(imgs)

            #imgs = normalize_latents(imgs,mean,std)

            imgs, noise = forward_cosine_noise(None, imgs, t, device= device)

            if np.random.random() <= 0.4:
                outputs = model(imgs, t)
            else:
                outputs = model(imgs,t,labels)

            loss = loss_fn(outputs, noise)

            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            loss_train += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Save model checkpoint with the current epoch in the filename

        with open("UNETAddEmbeddings/weights/celebA-diffusion-loss.txt", "a") as file:
            file.write(f"{loss_train / len(data_loader)}\n")

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(data_loader)))
        # Save model checkpoint every 20 epochs
        if epoch % 5 == 0:
            model_filename = f'UNETAddEmbeddings/weights/celebA-ldm-unet-epoch-{epoch}.pth'
            model_path = os.path.join('weights/', model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

        # Optional: Generate samples every 1 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                latent = reverse_diffusion(model,50, size=(32,32))
                #latent = unscale_latents(latent,mean,std)
                vae.decode(latent)
                random_tensor = torch.randint(0, 2, (1, 40), dtype=torch.float32)
                print("active classes: ", random_tensor)
                latent = reverse_diffusion_cfg(model,50,random_tensor,3)
                #latent = unscale_latents(latent,mean,std)
                vae.decode(latent)


                

if __name__ == '__main__':
    # Ensure weights directory exists
    os.makedirs('weights/', exist_ok=True)

    timesteps = 1000

    # Device Setup
    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    #Set up VAE
    vae_path = 'UNETAddEmbeddings/weights/celeb_vae_epoch_2.pth'
    vae_config = VAEConfig(z_dim=32,width=64, device=device)
    vae_model = VariationalAutoencoderGAN(vae_config).to(device)
    checkpoint = torch.load(vae_path)
    vae_model.load_state_dict(checkpoint['model_state_dict'])
    vae_model.eval()

    config = Config(c_in=32,width=64,num_classes=40)
    model = Unet(config)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss().to(device)
    print("param count: ", count_parameters(model))

    model_path = 'weights/celebA-ldm-unet-epoch-24.pth'
    epoch = 0


    #Dataloader
    csv_file = '/Users/ayanfe/Documents/Datasets/celebA/list_attr_celeba.csv'
    img_dir = '/Users/ayanfe/Documents/Datasets/celebA/img_align_celeba/img_align_celeba'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    subset_ratio = 1  # Use 50% of the dataset
    dataset = CelebADataset(csv_file, img_dir, transform=transform, subset_ratio=subset_ratio)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    #mean and std for latents precalculated 
    mean = torch.tensor([0.4947548273545867], device=device)
    
    std = torch.tensor([0.3398],device=device)

    '''
    # Optionally load model weights if needed
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    #run inference
    with torch.no_grad():
        latent = reverse_diffusion(model,50, size=(32,32))
        vae_model.decode(latent)
        random_tensor = torch.randint(0, 2, (1, 40), dtype=torch.float32)
        print("active classes: ", random_tensor)
        latent = reverse_diffusion_cfg(model,50,random_tensor,5)
        vae_model.decode(latent)
    '''
    
    training_loop(
        n_epochs=1000,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        device=device,
        data_loader=dataloader,
        vae = vae_model,
        timesteps=timesteps,
        epoch_start = epoch + 1,
        accumulation_steps= 1  # Adjust this value as needed
    )
    
    
    


