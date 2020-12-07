import torch
from torch import nn
import torchvision.utils as vutils

import numpy as np

from Unet.models import Encoder_U, Decoder_U

from Param import *
from utils import weights_init

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def fit(data, Encoder, Decoder, optimizer, criterion):

    img = data[0].to(device)
    
    encod_out, layers = Encoder(img)
    output = Decoder(layers)

    loss = criterion(output, img)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss.detach_()
    del output, layers, encod_out

    return loss



def trainUNet(dataloader, print_epoch=batch_size, verbose=True):

    
    assert image_size == 512

    Encoder_model = Encoder_U().to(device)
    Decoder_model = Decoder_U().to(device)

    
    if initialize_weights:
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
                    enc_out, layers = Encoder_model(data[0].to(device))
                    img_out = Decoder_model(layers).detach().cpu()
                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))
            

    return AE_losses, img_list, Encoder_model, Decoder_model




