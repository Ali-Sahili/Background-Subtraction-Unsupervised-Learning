import torch
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

from Param import device, method, Tensor
from torch.nn.utils import spectral_norm
import torch.nn as nn

# Plotting the Generator and Discriminator Loss During Training
def G_D_losses_plot(G_losses, D_losses):
    plt.figure(figsize=(10,5))
    plt.title(method + " -- Losses During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Plotting the Generator and Discriminator Loss During Training
def AE_losses_plot(AE_losses, name="reconstruction loss"):
    plt.figure(figsize=(10,5))
    plt.title(method + " -- Losses During Training")
    plt.plot(AE_losses,label=name)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# plot some real images and fake images side by side
def Results_plot(dataloader, img_list):

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(2,1,1)
    plt.axis("off")
    plt.title(method + " -- Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:10].to(device), nrow=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(2,1,2)
    plt.axis("off")
    plt.title(method + " -- Reconstructed Images")
    #plt.imshow(np.transpose(vutils.make_grid(img_list[0][:10].to(device), nrow=5, normalize=True).cpu(),(1,2,0)))
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()



# plot some real images and fake images side by side
def Results_plot_test(data, img_list, image_size):

    plt.figure(figsize=(15,15))
    # Plot the real images from the last epoch
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title(method + " -- Reconstructed Images")
    plt.imshow(np.transpose(data[-1],(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title(method + " -- Reconstructed Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
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



def Covariance_Correlation(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]

    # compute the covariance matrix
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())

    # compute the correlation matrix
    inv = torch.inverse(torch.diag(torch.diag(cov)).sqrt())
    cor = inv@cov@inv

    return cov, cor



# save model of AAE
def save_model(model, filename):
    print('Best model so far, saving it...')
    torch.save(model.state_dict(), filename)  # filename example: conv_autoencoder.pth 


# custom weights initialization called on netG and netE
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

"""
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
"""



# spectral normalization
def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m

# Compute noise component of the image --> foreground objects
def soft_threshold(lamda, softIn): # softIn = Input - output

    th = float(lamda) / 2.0

    if (lamda == 0):
        return softIn

    x = torch.zeros(softIn.shape).to(device)


    k = softIn > th
    x[k] = softIn[k] - th

    k = torch.abs(softIn) <= th
    x[k] = 0

    k = softIn < -th
    x[k] = softIn[k] + th
    
    #x = torch.from_numpy(x).to(device)

    return x





