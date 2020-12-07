import torch
from torch import nn
import torchvision.utils as vutils

import numpy as np

from VAE.VAE_Attention import VAE_Attention_model

from Param import *
from utils import weights_init
from VAE.Loss_functions import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def fit(data1, data2, VAE_model, optimizer):

    img1 = data1[0].to(device)
    img2 = data2[0].to(device) 

    img = torch.cat((img1,img2),1)
  
    output, mu, logvar = VAE_model(img)

    loss = loss_function2(output, img1, mu, logvar)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss



def trainVAEWA(dataloader1, dataloader2, print_epoch=batch_size, verbose=True):

    
    if image_size_W == image_size_H:
        image_size = image_size_H

    if method == 'VAE_Attention':
        if image_size == 512:
            VAE_model = VAE_Attention_model().to(device)
    else:
        assert(0)
    
    if initialize_weights:
        VAE_model.apply(weights_init)


    optimizer = torch.optim.Adam(list(VAE_model.parameters()), lr=lr, weight_decay=1e-5)

    print("Starting Training Loop...")


    AE_losses = []
    img_list = []

    VAE_model.train()
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data1,data2) in enumerate(zip(dataloader1, dataloader2), 0):
         
            recons_loss = fit(data1, data2, VAE_model, optimizer)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader1),
                           recons_loss.item()))


            # Save Losses for plotting later
            AE_losses.append(recons_loss.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader1)-1)):
                VAE_model.eval()
                with torch.no_grad():
                        img_out,_,_ = VAE_model(torch.cat((data1[0],data2[0]),1).to(device))
                        img_out = img_out.detach().cpu()

                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))
                VAE_model.train()

                """
                n = min(data1.shape[0], 8)
                result = torch.cat([data1[:n],
                                      img_out[:n]])
                vutils.save_image(result,   # .cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                """

    return AE_losses, img_list, VAE_model



