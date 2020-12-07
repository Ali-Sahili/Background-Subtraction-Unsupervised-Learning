import torch
from torch import nn
import torchvision.utils as vutils

import numpy as np


from AE_swish.Encoders import *
from AE_swish.Decoders import *

from Param import *
from utils import Covariance_Correlation, weights_init, add_sn



from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def fit(data, Encoder, Decoder, optimizer, criterion):

    img = data[0].to(device)
    
    if Initializze_BG:
        encod_out, bg = Encoder(img)
    else:
        encod_out = Encoder(img)

    output = Decoder(encod_out)

    if denoising_autoencoder:
        encod_out = Encoder(output)
        output = Decoder(encod_out)
     
    if Initializze_BG:
        loss = criterion(output, img) + criterion(output, bg)
    else:
        loss = criterion(output, img)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss.detach_()

    return loss




def trainAE_swish(dataloader, print_epoch=batch_size, verbose=True):

    
    if image_size_W == image_size_H:
        image_size = image_size_H

    if image_size == 512:
            Encoder_model = Encoder512_swish().to(device)
            Decoder_model = Decoder512_swish().to(device)
    elif image_size == 256:
            Encoder_model = Encoder256_swish().to(device)
            Decoder_model = Decoder256_swish().to(device)
    else:
            assert(0)

    
    if initialize_weights:
            Encoder_model.apply(weights_init)
            Decoder_model.apply(weights_init)
            print('spectral normalization ...')
            Encoder_model.apply(add_sn)
            Decoder_model.apply(add_sn)

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
                    if Initializze_BG: enc_out, bg = Encoder_model(data[0].to(device))
                    else: enc_out = Encoder_model(data[0].to(device))

                    if verbose: print('latent space: ',enc_out.shape)
                    img_out = Decoder_model(enc_out).detach().cpu()
                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))
            

    return AE_losses, img_list, Encoder_model, Decoder_model




