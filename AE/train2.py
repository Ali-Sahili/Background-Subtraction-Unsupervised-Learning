import torch
from torch import nn
import torchvision.utils as vutils

import numpy as np

from models.patchmodels_Attention import Encoder_Decoder512
from models.AE_Attention import AE_Attention

from Param import *
from utils import weights_init

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def fit(data1, data2, AE_model, optimizer, criterion):

    img1 = data1[0].to(device)
    img2 = data2[0].to(device) 

    img = torch.cat((img1,img2),1)
  
    output = AE_model(img)

    if denoising_autoencoder:
        output = AE_model(output)

    loss = criterion(output, img1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss



def trainPWA(dataloader1, dataloader2, print_epoch=batch_size, verbose=True):

    
    if image_size_W == image_size_H:
        image_size = image_size_H

 
    if image_size == 512:
        if method == 'patchLevel_withAttention':
            AE_model = Encoder_Decoder512().to(device)
        elif method == 'AE_Attention':
            AE_model = AE_Attention().to(device)
    else:
        assert(0)
    
    if initialize_weights:
        AE_model.apply(weights_init)
 

    if loss_ == True:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()  # first choice
    #criterion = nn.SmoothL1Loss() # second choice
    #criterion = LogCoshLoss() # third choice
    #criterion = XTanhLoss()  # fourth choice
    #criterion = XSigmoidLoss()  # fifth choice
    #criterion = AlgebraicLoss()  # sixth choice


    optimizer = torch.optim.Adam(list(AE_model.parameters()), lr=lr, weight_decay=1e-5)

    print("Starting Training Loop...")


    AE_losses = []
    img_list = []

    AE_model.train()
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data1,data2) in enumerate(zip(dataloader1, dataloader2), 0):
         
            recons_loss = fit(data1, data2, AE_model, optimizer, criterion)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader1),
                           recons_loss.item()))


            # Save Losses for plotting later
            AE_losses.append(recons_loss.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader1)-1)):
                AE_model.eval()
                with torch.no_grad():
                        img_out = AE_model(torch.cat((data1[0],data2[0]),1).to(device)).detach().cpu()

                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))
                AE_model.train()

                """
                n = min(data1.shape[0], 8)
                result = torch.cat([data1[:n],
                                      img_out[:n]])
                vutils.save_image(result,   # .cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                """

    return AE_losses, img_list, AE_model



