import torch
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

from Param import device


# plot some real images and fake images side by side
def Results_plot(dataloader, BG_list, FG_list):

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:10].to(device), nrow=5, normalize=True).cpu(),(1,2,0)))

    # Plot the background images
    plt.subplot(3,1,2)
    plt.axis("off")
    plt.title("Reconstructed Images")
    #plt.imshow(np.transpose(vutils.make_grid(img_list[0][:10].to(device), nrow=5, normalize=True).cpu(),(1,2,0)))
    plt.imshow(np.transpose(BG_list[-1],(1,2,0)))

    # Plot the foreground images
    plt.subplot(3,1,3)
    plt.axis("off")
    plt.title("Reconstructed Images")
    #plt.imshow(np.transpose(vutils.make_grid(img_list[0][:10].to(device), nrow=5, normalize=True).cpu(),(1,2,0)))
    plt.imshow(np.transpose(FG_list[-1],(1,2,0)))
    plt.show()


# Plot some input data
def Data_plot(dataloader, image_size):
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:image_size], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()




