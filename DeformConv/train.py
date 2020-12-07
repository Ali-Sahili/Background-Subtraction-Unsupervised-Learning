import torch
from torch import nn
import torchvision.utils as vutils

import numpy as np

from DeformConv.model import *

from Param import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def fit(data, Encoder, Decoder, optimizer, criterion, max_norm=0):

    img = data[0].to(device)
    
    encod_out = Encoder(img)
    output = Decoder(encod_out)

    loss = criterion(img, output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss.detach_()

    if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm)

    return loss

def trainAE(dataloader, print_epoch=batch_size, verbose=True):

    
    assert image_size == 256
    Encoder_model = DeformConvNet_Encoder256().to(device)
    Decoder_model = Decoder256().to(device)

    criterion = nn.MSELoss()  

    n_parametersE = Encoder_model.nb_parameters()
    n_parametersD = sum(p.numel() for p in Decoder_model.parameters() if p.requires_grad)
    print('number of params in encoder:', n_parametersE)
    print('number of params in decoder:', n_parametersD)
    optimizer = torch.optim.Adam(list(Encoder_model.parameters())+list(Decoder_model.parameters()), lr=lr)
    #optimizer = torch.optim.AdamW(list(Encoder_model.parameters())+list(Decoder_model.parameters()), lr=lr)

    print("Starting Training Loop...")


    AE_losses = []
    img_list = []
    # For each epoch
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
         
            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)

            recons_loss = fit(data, Encoder_model, Decoder_model, optimizer, criterion)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader), recons_loss.item()))

            # Save Losses for plotting later
            AE_losses.append(recons_loss.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    enc_out = Encoder_model(data[0].to(device))
                    img_out = Decoder_model(enc_out).detach().cpu()
                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))
            

    return AE_losses, img_list, Encoder_model, Decoder_model




