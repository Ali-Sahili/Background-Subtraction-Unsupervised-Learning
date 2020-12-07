import torch
from torch import nn
import torchvision.utils as vutils
import pytorch_ssim

import numpy as np

from models.Encoders import *
from models.Decoders import *
from models.patchmodels import Encoder512_F, Decoder512_F
from models.patchmodels_Attention import Encoder_Decoder512
from MultiScale_input.model import FinalModel

from Param import *
from utils import Covariance_Correlation, weights_init

from Losses import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def fit(data, Encoder, Decoder, optimizer, criterion, max_norm=0):

    img = data[0].to(device)
    
    if Initializze_BG:
        encod_out, bg = Encoder(img)
    else:
        encod_out = Encoder(img)

    output = Decoder(encod_out)

    #m = torch.mean(img)
    #m_out = torch.mean(output_tmp)
    #m_ = m - m_out
    #output = output_tmp + m_

    if denoising_autoencoder:
        encod_out = Encoder(output)
        output = Decoder(encod_out)
     
    if Initializze_BG:
        loss = criterion(output, img) + criterion(output, bg)
    else:
        loss = criterion(output, img)

    #loss_1 = criterion(output, img)
    #loss_2 = 1 - pytorch_ssim.SSIM()(output, img)

    #k = 0.2
    #loss = (1-k)*loss_1 + k*loss_2


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss.detach_()

    if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm)

    return loss


def fit2(data, AE_model, optimizer, criterion):

    img = data[0].to(device)
      
    output = AE_model(img)

    if denoising_autoencoder:
        output = AE_model(output)

    loss = criterion(output, img)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss.detach_()

    return loss

def trainAE(dataloader, print_epoch=batch_size, verbose=True):

    
    if image_size_W == image_size_H:
        image_size = image_size_H

    if method == 'patchLevel_withoutAttention':
        if image_size == 512:
            Encoder_model = Encoder512_F().to(device)
            Decoder_model = Decoder512_F().to(device)
        else:
            assert(0)

    elif method == 'MultiScale_input':
        if image_size == 512:
            AE_model = FinalModel().to(device)

    elif method == 'Autoencoder' or method == 'Couple_Autoencoders':
            if image_size == 64:
                Encoder_model = Encoder64().to(device)
                Decoder_model = Decoder64().to(device)
            elif image_size == 128:
                Encoder_model = Encoder128().to(device)
                Decoder_model = Decoder128().to(device)
            elif image_size == 256:
                Encoder_model = Encoder256().to(device)
                Decoder_model = Decoder256().to(device)
            elif image_size == 512:
                Encoder_model = Encoder512().to(device)
                Decoder_model = Decoder512().to(device)
            elif image_size_W == 1280 and image_size_H == 720:
                Encoder_model = Encoder1280().to(device)
                Decoder_model = Decoder1280().to(device)
            else:
                assert(0)

    elif method == 'AE_GroupNorm':
            if image_size == 512:
                Encoder_model = Encoder512_G().to(device)
                Decoder_model = Decoder512_G().to(device)

    else:
        assert(0)
    
    if initialize_weights:
        if method == 'MultiScale_input':
            AE_model.apply(weights_init)
        else:
            Encoder_model.apply(weights_init)
            Decoder_model.apply(weights_init)

    if loss_ == True:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()  # first choice
        #criterion = nn.SmoothL1Loss() # second choice
        #criterion = LogCoshLoss() # third choice
        #criterion = XTanhLoss()  # fourth choice
        #criterion = XSigmoidLoss()  # fifth choice
        #criterion = AlgebraicLoss()  # sixth choice
        #criterion = PhotometricLoss() # seventh choice

    if method == 'MultiScale_input':
        optimizer = torch.optim.Adam(list(AE_model.parameters()), lr=lr, weight_decay=1e-5)
    else:
        n_parametersE = sum(p.numel() for p in Encoder_model.parameters() if p.requires_grad)
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

            if method == 'MultiScale_input':
                #AE_model.train()
                recons_loss = fit2(data, AE_model, optimizer, criterion)
            else:
                #Encoder_model.train()
                #Decoder_model.train()
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
                    if method == 'MultiScale_input':
                        #AE_model.eval()
                        img_out = AE_model(data[0].to(device)).detach().cpu()
                    else:
                        #Encoder_model.eval()
                        #Decoder_model.eval()

                        if Initializze_BG: enc_out, bg = Encoder_model(data[0].to(device))
                        else: enc_out = Encoder_model(data[0].to(device))

                        if verbose: print('latent space: ',enc_out.shape)
                        img_out = Decoder_model(enc_out).detach().cpu()
                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))
            

    if method == 'MultiScale_input':
        return AE_losses, img_list, AE_model
    else:
        return AE_losses, img_list, Encoder_model, Decoder_model




