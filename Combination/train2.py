"""  Use auto-encoder models (convolutif AE, patch-level AE, with different input size) using two input combinations: RGB image and its optical flow map.
Each of two inputs have been fed to an ecoder and the result is fused at the latent space and apply one decoder to decode the information, 
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

def fit(data1, data2, Encoder_img, Encoder_flow, Decoder, optimizer_img, optimizer_flow, criterion):

    img1 = data1[0].to(device)
    img2 = data2[0].to(device) 

    encod_out1 = Encoder_img(img1)
    encod_out2 = Encoder_flow(img2)

    encod_out = torch.cat((encod_out1, encod_out2), 1)

    output = Decoder(encod_out)
    
    e = 0.001*torch.ones(output.shape).to(device)
    #loss_tensor = ((output-img2)**2 + e**2) #* (mask)  # L2-norm
    loss_tensor = torch.sqrt((output-img2)**2 + e**2) #* (mask)  # L1-norm
    loss = torch.sum(loss_tensor)
    loss /= img2.shape[2]*img2.shape[3]*img2.shape[1]*img2.shape[0]

    optimizer_img.zero_grad()
    optimizer_flow.zero_grad()
    loss.backward()
    optimizer_img.step()
    optimizer_flow.step()

    return loss




def train2E1D(dataloader1, dataloader2, print_epoch=batch_size, verbose=True):

    assert len(dataloader1) == len(dataloader2), "two datasets should have the same size."
    
    
    if image_size_W == image_size_H:
        image_size = image_size_H

    if patch_mode:
        if image_size == 512:
            Encoder_img = Encoder512_F(nc=nc).to(device)
            Encoder_flow = Encoder512_F(nc=nc_2).to(device)
            Decoder_model = Decoder512_F(nc=nc_2, nz=2*nz).to(device)
        else:
            assert(0)

    else:
            if image_size == 64:
                Encoder_img = Encoder64(nc=nc).to(device)
                Encoder_flow = Encoder64(nc=nc_2).to(device)
                Decoder_model = Decoder64(nc=nc_2, nz=2*nz).to(device)
            elif image_size == 128:
                Encoder_img = Encoder128(nc=nc).to(device)
                Encoder_flow = Encoder128(nc=nc_2).to(device)
                Decoder_model = Decoder128(nc=nc_2, nz=2*nz).to(device)
            elif image_size == 256:
                Encoder_img = Encoder256(nc=nc).to(device)
                Encoder_flow = Encoder256(nc=nc_2).to(device)
                Decoder_model = Decoder256(nc=nc_2, nz=2*nz).to(device)
            elif image_size == 512:
                Encoder_img = Encoder512(nc=nc).to(device)
                Encoder_flow = Encoder512(nc=nc_2).to(device)
                Decoder_model = Decoder512(nc=nc_2, nz=2*nz).to(device)

            else:
                assert(0)

    
    if initialize_weights:
        Encoder_img.apply(weights_init)
        Encoder_flow.apply(weights_init)
        Decoder_model.apply(weights_init)


    if loss_:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss() 

    optimizer_img = torch.optim.Adam(list(Encoder_img.parameters())+list(Decoder_model.parameters()), lr=lr, weight_decay=1e-5)
    optimizer_flow = torch.optim.Adam(list(Encoder_flow.parameters())+list(Decoder_model.parameters()), lr=lr, weight_decay=1e-5)

    print("Starting Training Loop...")


    AE_losses = []
    img_list = []
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data1,data2) in enumerate(zip(dataloader1, dataloader2), 0):
         
            if verbose: print(data1[0].shape);print(data2[0].shape);
            if verbose: print(data1[1].shape);print(data2[1].shape);

            recons_loss = fit(data1, data2, Encoder_img, Encoder_flow, Decoder_model, optimizer_img, optimizer_flow, criterion)

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
                    enc_out1 = Encoder_img(data1[0].to(device))
                    enc_out2 = Encoder_flow(data2[0].to(device))

                    img_out = Decoder_model(torch.cat((enc_out1, enc_out2),1)).detach().cpu()
                    imgs_out = img_out.permute(1,0,2,3)[0:3].permute(1,0,2,3)
                    #print(imgs_out.shape)
                img_list.append(vutils.make_grid(imgs_out[0:10], nrow=5, normalize=True))
            


    return AE_losses, img_list, Encoder_img, Encoder_flow, Decoder_model





