import torch
from torch import nn

import torchvision.utils as vutils

import numpy as np

from models.Encoders import *
from models.Decoders import *

from background_initialization import Initialize

from Param import *
from utils import Covariance_Correlation, weights_init

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def fit(data, Encoder_model_1, Decoder_model_1, Encoder_model_2, Decoder_model_2, optimizer_1, optimizer_2, criterion):

    img = data[0].to(device)


    #ini_model = Initialize().to(device)
    #background, foreground = ini_model(img)

    if Initializze_BG:
        encod_out_1, bg = Encoder_model_1(img)
        output_1 = Decoder_model_1(encod_out_1)
        loss_1 = criterion(img, output_1) + 0.5*criterion(bg, output_1)
    else:
        encod_out_1 = Encoder_model_1(img)
        output_1 = Decoder_model_1(encod_out_1)
        loss_1 = criterion(img, output_1)

    optimizer_1.zero_grad()    
    loss_1.backward()
    optimizer_1.step()

 
    optimizer_2.zero_grad()

    if Initializze_BG:
        encod_out_2, bg = Encoder_model_2(output_1.detach())
    else:
        encod_out_2 = Encoder_model_2(output_1.detach())

    output_2 = Decoder_model_2(encod_out_2)
    
    loss_2 = criterion(output_1.detach(), output_2)

    loss_2.backward()
    optimizer_2.step()


    return loss_1, loss_2



def trainCAE(dataloader, print_epoch=batch_size, verbose=True):

    
    if image_size_W == image_size_H:
        image_size = image_size_H

        # Create encoder and decoder models
        if image_size == 64:
            Encoder_model_1 = Encoder64().to(device)
            Decoder_model_1 = Decoder64().to(device)

            Encoder_model_2 = Encoder64().to(device)
            Decoder_model_2 = Decoder64().to(device)

        elif image_size == 128:
            Encoder_model_1 = Encoder128().to(device)
            Decoder_model_1 = Decoder128().to(device)

            Encoder_model_2 = Encoder128().to(device)
            Decoder_model_2 = Decoder128().to(device)

        elif image_size == 256:
            Encoder_model_1 = Encoder256().to(device)
            Decoder_model_1 = Decoder256().to(device)

            Encoder_model_2 = Encoder256().to(device)
            Decoder_model_2 = Decoder256().to(device)

        elif image_size == 512:
            Encoder_model_1 = Encoder512().to(device)
            Decoder_model_1 = Decoder512().to(device)

            Encoder_model_2 = Encoder512().to(device)
            Decoder_model_2 = Decoder512().to(device)

        else:
            assert(0)

    elif image_size_W == 1280 and image_size_H == 720:
        Encoder_model_1 = Encoder1280().to(device)
        Decoder_model_1 = Decoder1280().to(device)

        Encoder_model_2 = Encoder1280().to(device)
        Decoder_model_2 = Decoder1280().to(device)

    else:
        assert(0)

    
    if initialize_weights:
        Encoder_model_1.apply(weights_init)
        Decoder_model_1.apply(weights_init)

        Encoder_model_2.apply(weights_init)
        Decoder_model_2.apply(weights_init)
    

    if loss_:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss() 


    optimizer_1 = torch.optim.Adam(list(Encoder_model_1.parameters())+list(Decoder_model_1.parameters()), lr=lr, weight_decay=1e-5)
    optimizer_2 = torch.optim.Adam(list(Encoder_model_2.parameters())+list(Decoder_model_2.parameters()), lr=lr, weight_decay=1e-5)

    print("Starting Training Loop...")


    AE_losses_1 = []
    AE_losses_2 = []
    img_list1 = []
    img_list2 = []
    # For each epoch
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        # For each batch in the dataloader
        Encoder_model_1.train()
        Decoder_model_1.train()
                    
        Encoder_model_2.train()
        Decoder_model_2.train()
        for i, data in enumerate(dataloader, 0):
         
            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)

            recons_loss_1, recons_loss_2 = fit(data, Encoder_model_1, Decoder_model_1, Encoder_model_2, Decoder_model_2, optimizer_1, optimizer_2, criterion)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f/%.4f'
                        % (epoch+1, num_epochs, i, len(dataloader),
                           recons_loss_1.item(), recons_loss_2.item()))

            # Save Losses for plotting later
            AE_losses_1.append(recons_loss_1.item())
            AE_losses_2.append(recons_loss_2.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                Encoder_model_1.eval()
                Decoder_model_1.eval()
                    
                Encoder_model_2.eval()
                Decoder_model_2.eval()

                with torch.no_grad():

                    if Initializze_BG: enc_out, bg = Encoder_model_1(data[0].to(device))
                    else: enc_out = Encoder_model_1(data[0].to(device))
                    out1 = Decoder_model_1(enc_out)

                    if Initializze_BG: enc_out, bg = Encoder_model_2(out1)
                    else: enc_out = Encoder_model_2(out1)
                    img_out = Decoder_model_2(enc_out)
                img_list1.append(vutils.make_grid(out1.detach().cpu()[0:10], nrow=5, normalize=True))
                img_list2.append(vutils.make_grid(img_out.detach().cpu()[0:10], nrow=5, normalize=True))



    return AE_losses_1, AE_losses_2, img_list1, img_list2, Encoder_model_1, Encoder_model_2, Decoder_model_1, Decoder_model_2








