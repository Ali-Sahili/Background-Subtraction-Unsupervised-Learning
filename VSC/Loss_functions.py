import torch
from torch import nn
import torch.nn.functional as F

#import pytorch_ssim
from Param import loss_

# This function is used in the variational auto-encoder
def loss_function(x, recon_x, mu, logvar, logspike):
        
    alpha = 0.5
    beta = 0.1

    # Reconstruction term sum (mean?) per batch
    assert x.shape == recon_x.shape
    #flat_input_sz = x.shape[1] * x.shape[2] * x.shape[3]
    #BCE = F.binary_cross_entropy(recon_x.view(-1, flat_input_sz), x.view(-1, flat_input_sz),
                                         #size_average = False)


    criterion = nn.MSELoss()
    recons_loss = criterion(x, recon_x)    
   
    # see Appendix B from VSC paper / Formula 6
    spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 

    prior1 = -0.5 * torch.sum(spike.mul(1 + logvar - mu.pow(2) - logvar.exp()))
    prior21 = (1 - spike).mul(torch.log((1 - spike) / (1 - alpha)))
    prior22 = spike.mul(torch.log(spike / alpha))
    prior2 = torch.sum(prior21 + prior22)
    PRIOR = prior1 + prior2

    #LOSS = BCE + beta * PRIOR
    LOSS = recons_loss + beta * PRIOR

    return LOSS, recons_loss

