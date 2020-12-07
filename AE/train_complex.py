import torch
from torch import nn
import torchvision.utils as vutils
import pytorch_ssim

import numpy as np

from AE.Complex_Attention import *

from Param import *
from utils import weights_init

from Losses import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True




def train(dataloader, print_epoch=batch_size, verbose=False):

    assert image_size == 256 or image_size == 128

    nc = 3
    h = image_size  
    w = image_size
    dim_encoding = 6
    nheads = 8
    nce = 8
    nzf = 50   
    nze = 2  
    ncg = 8
    ncl = 8
    mask_head_flag = False
    masked_head = None
    model = Complex_Attention_Autoencoder(nze,nzf,nc, nce,nheads, h, w, dim_encoding,ncg,ncl,  
                               mask_head_flag,masked_head).to(device)



    
    if initialize_weights:
        model.apply(weights_init)

    criterion = nn.MSELoss()  

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("Starting Training Loop...")


    AE_losses = []
    img_list = []
    att_enc_list = []
    att_dec_list = []
    max_norm = 0
    # For each epoch
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
         
            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)

            img = data[0].to(device)
    
            encod_out, attention_maps_encoder = model(img)
            output, attention_maps_decoder = model(encod_out)

            loss = criterion(output, img)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss.detach_()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader), loss.item()))

            # Save Losses for plotting later
            AE_losses.append(loss.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    enc_out, att_enc = model(data[0].to(device))
                    img_out, att_dec = model(enc_out)
                img_list.append(vutils.make_grid(img_out.detach().cpu()[0:10,:3], nrow=5, normalize=True))
                att_enc_list.append(vutils.make_grid(att_enc.detach().cpu()[0:10,:3], nrow=5, normalize=True))
                att_dec_list.append(vutils.make_grid(att_dec.detach().cpu()[0:10,:3], nrow=5, normalize=True))
            

    return AE_losses, img_list, att_enc_list, att_dec_list, model




