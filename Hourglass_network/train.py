import torch
from torch import nn
import torchvision.utils as vutils

import numpy as np

from focal_loss import FocalLoss
from Param import *
from utils import weights_init

from net import PoseNet


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def fit(data, mask, Net, optimizer, criterion, max_norm=0):

    img = data[0].to(device)
    
    heat_maps, output = Net(img)

    loss = 0
    for i in range(output.shape[1]):
        loss += criterion(output[:,i], mask[0].to(device))


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss.detach_()

    if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm)

    return loss


def train(dataloader, dataloader_mask, print_epoch=batch_size, verbose=False):

    
    assert image_size == 256

    model = PoseNet(nstack, image_size, oup_dim, bn, increase).to(device)

    
    #if initialize_weights:
     #   model.apply(weights_init)

    #criterion = nn.MSELoss()  
    criterion = FocalLoss(gamma=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params ', n_parameters)


    print("Starting Training Loop...")


    losses = []
    img_list = []
    heat_maps_list = []
    # For each epoch
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()

        # For each batch in the dataloader
        for i, (data, mask) in enumerate(zip(dataloader, dataloader_mask), 0):
         
            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)

            recons_loss = fit(data, mask, model, optimizer, criterion)

            # Output training stats
            if i % print_epoch == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader), recons_loss.item()))

            # Save Losses for plotting later
            losses.append(recons_loss.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    heat_maps, img_out = model(data[0].to(device))
                    img_out = img_out.detach().cpu()
                    heat_maps = heat_maps.detach().cpu()

                img_list.append(vutils.make_grid(img_out[0:10,0], nrow=5, normalize=True))
        
        if epoch == (num_epochs-1):   
            for qq in range(heat_maps.shape[2]):
                heat_maps_list.append(vutils.make_grid(heat_maps[0:5,nstack-1,qq].unsqueeze(1), nrow=5, normalize=True, padding=5, pad_value=1).permute(1,2,0))
            heat_map_out = np.vstack(heat_maps_list)

    return losses, img_list, heat_map_out, model





