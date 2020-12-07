import torch
from torch import nn
import torchvision.utils as vutils

import numpy as np

from Atrous_Convolution.Encoders import Encoder256_Atrous, Encoder256_Atrous_fuse
from Atrous_Convolution.Decoders import Decoder256

from Param import *
from utils import Covariance_Correlation, weights_init

from Losses import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def fit(data, Encoder, Decoder, optimizer, criterion):

    img = data[0].to(device)

    encod_out = Encoder(img)

    output = Decoder(encod_out)
     
    loss = criterion(output, img)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss.detach_()

    del output

    return loss



def train_atrousAE(dataloader, print_epoch=batch_size, verbose=True):

    
    assert image_size == 256


    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if method == 'atrous_AE': Encoder_model = Encoder256_Atrous().to(device) 
    elif method == 'atrous_AE_fuse': Encoder_model = Encoder256_Atrous_fuse().to(device) 
    Decoder_model = Decoder256().to(device)
    
    Encoder_model.apply(weights_init)
    Decoder_model.apply(weights_init)

    if loss_ == True:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()  

    optimizer = torch.optim.Adam(list(Encoder_model.parameters())+list(Decoder_model.parameters()), lr=lr)

    print("Starting Training Loop...")


    AE_losses = []
    img_list = []
    # For each epoch
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        Encoder_model.train()
        Decoder_model.train()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            

            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)

            recons_loss = fit(data, Encoder_model, Decoder_model, optimizer, criterion)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader),
                           recons_loss.item()))

            # Save Losses for plotting later
            AE_losses.append(recons_loss.item())
            
        # Check how the generator is doing by saving G's output on fixed_noise
        if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                    Encoder_model.eval()
                    Decoder_model.eval()

                    enc_out = Encoder_model(data[0].to(device))
                    img_out = Decoder_model(enc_out).detach().cpu()

            img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))
            

    return AE_losses, img_list, Encoder_model, Decoder_model




