import torch
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

from Param import device
import torch.nn as nn


# Plotting the Generator and Discriminator Loss During Training
def AE_losses_plot(AE_losses, name="reconstruction loss"):
    plt.figure(figsize=(10,5))
    plt.title("Losses During Training")
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
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:10].to(device), nrow=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(2,1,2)
    plt.axis("off")
    plt.title("Reconstructed Images")
    #plt.imshow(np.transpose(vutils.make_grid(img_list[0][:10].to(device), nrow=5, normalize=True).cpu(),(1,2,0)))
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()


# Plot some input data
def heatmap_plot(img_list):
    # Plot some training images
    plt.figure(figsize=(15,15))
    plt.axis("off")
    plt.title("Heat maps")
    plt.imshow(img_list)
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






