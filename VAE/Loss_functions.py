import torch
from torch import nn

#import pytorch_ssim
from Param import loss_

# This function is used in the variational auto-encoder

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    if loss_:
        reconstruction_function = nn.BCELoss()
    else:
        reconstruction_function = nn.MSELoss() 


    mse_loss = reconstruction_function(recon_x, x)  # mse loss
    #ssim_loss = 1 - pytorch_ssim.SSIM()(recon_x, x) # ssim loss

    #k = 0.2
    #BCE = (1-k)*mse_loss + k*ssim_loss

    BCE = mse_loss

    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD




def loss_function2(recon_x, x, mu, logvar):
    beta = 0.8
    SparsityLoss = nn.L1Loss()

    L1loss = SparsityLoss(recon_x, x)
    KLD = -0.5 * beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return L1loss + KLD
