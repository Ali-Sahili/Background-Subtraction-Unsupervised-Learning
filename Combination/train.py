"""  Use auto-encoder models (convolutif AE, patch-level AE, with different input size) using two input combinations: RGB image and its optical flow map.
These two inputs have been fused on the channel dimension to get a tensor of b_size x nc1+nc2 x H x W
where nc1 is the number of image channel (nc1=3) and nc2 is the number of channels of the optical flow map (nc2=1 or 3).
At the end, the ouput is an optical flow map compared to the input optical flow map.

"""


import torch
from torch import nn
import torchvision.utils as vutils

import numpy as np

from models.Encoders import *
from models.Decoders import *

from Param import *
from utils import weights_init


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def fit(data1, data2, Encoder, Decoder, optimizer, criterion):

    img1 = data1[0].to(device)
    img2 = data2[0].to(device) 

    img = torch.cat((img1,img2),1)  #  channels = channels of data1 + channels of data2

    if Initializze_BG:
        encod_out, bg = Encoder(img)
    else:
        encod_out = Encoder(img)

    output = Decoder(encod_out)
    
    e = 0.001*torch.ones(output.shape).to(device)
    #loss_tensor = ((output-img2)**2 + e**2) #* (mask)  # L2-norm
    loss_tensor = torch.sqrt((output-img1)**2 + e**2) #* (mask)  # L1-norm
    loss = torch.sum(loss_tensor)
    loss /= img.shape[2]*img.shape[3]*img.shape[1]*img.shape[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    return loss




def train1E1D(dataloader1, dataloader2, print_epoch=batch_size, verbose=True):

    assert len(dataloader1) == len(dataloader2), "two datasets should have the same size."
    
    
    if image_size_W == image_size_H:
        image_size = image_size_H

    if method == 'patchLevel_withoutAttention':
        if image_size == 512:
            Encoder_model = Encoder512_F(nc=nc+nc_2).to(device)
            Decoder_model = Decoder512_F(nc=nc_2).to(device)
        else:
            assert(0)

    elif method == 'patchLevel_withAttention':
        if image_size == 512:
            AE_model = Encoder_Decoder512(nc=nc+nc_2).to(device)


    elif method == 'Autoencoder' or method == 'Couple_Autoencoders':
            if image_size == 64:
                Encoder_model = Encoder64(nc=nc+nc_2).to(device)
                Decoder_model = Decoder64(nc=nc_2).to(device)
            elif image_size == 128:
                Encoder_model = Encoder128(nc=nc+nc_2).to(device)
                Decoder_model = Decoder128(nc=nc_2).to(device)
            elif image_size == 256:
                Encoder_model = Encoder256(nc=nc+nc_2).to(device)
                Decoder_model = Decoder256(nc=nc_2).to(device)
            elif image_size == 512:
                Encoder_model = Encoder512(nc=nc+nc_2).to(device)
                Decoder_model = Decoder512(nc=nc_2).to(device)
            elif image_size_W == 1280 and image_size_H == 720:
                Encoder_model = Encoder1280(nc=nc+nc_2).to(device)
                Decoder_model = Decoder1280(nc=nc_2).to(device)
            else:
                assert(0)

    else:
        if image_size == 512:
            Encoder_model = Encoder512(nc=nc+nc_2).to(device)
            Decoder_model = Decoder512(nc=nc_2).to(device)
        else:
            assert(0)
    
    if initialize_weights:
        Encoder_model.apply(weights_init)
        Decoder_model.apply(weights_init)


    if loss_:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss() 
    #criterion = nn.SmoothL1Loss() # second choice
    #criterion = LogCoshLoss() # third choice
    #criterion = XTanhLoss()  # fourth choice
    #criterion = XSigmoidLoss()  # fifth choice
    #criterion = AlgebraicLoss()  # sixth choice

    optimizer = torch.optim.Adam(list(Encoder_model.parameters())+list(Decoder_model.parameters()), lr=lr, weight_decay=1e-5)

    print("Starting Training Loop...")
    Encoder_model.train()
    Decoder_model.train()

    AE_losses = []
    img_list = []
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data1,data2) in enumerate(zip(dataloader1, dataloader2), 0):
         
            if verbose: print(data1[0].shape);print(data2[0].shape);
            if verbose: print(data1[1].shape);print(data2[1].shape);

            recons_loss = fit(data1, data2, Encoder_model, Decoder_model, optimizer, criterion)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader1),
                           recons_loss.item()))

            # Save Losses for plotting later
            AE_losses.append(recons_loss.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader1)-1)):
                with torch.no_grad():
                    if Initializze_BG: enc_out, bg = Encoder_model(torch.cat((data1[0],data2[0]),1).to(device))
                    else: enc_out = Encoder_model(torch.cat((data1[0],data2[0]),1).to(device))
                    if verbose: print('latent space: ',enc_out.shape)
                    img_out = Decoder_model(enc_out).detach().cpu()
                    imgs_out = img_out.permute(1,0,2,3)[0:3].permute(1,0,2,3)
                    #print(imgs_out.shape)
                img_list.append(vutils.make_grid(imgs_out[0:10], nrow=5, normalize=True))
            


    return AE_losses, img_list, Encoder_model, Decoder_model





