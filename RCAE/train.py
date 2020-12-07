import torch
from torch import nn
import torchvision.utils as vutils

import numpy as np
import time 
import gc 

from models.Encoders import *
from models.Decoders import *
from models.patchmodels import Encoder512_F, Decoder512_F
from models.patchmodels_Attention import Encoder_Decoder512

from Param import *
from utils import Covariance_Correlation, weights_init

from Losses import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def soft_threshold(lamda, softIn): # softIn = Input - output

    th = float(lamda) / 2.0

    if (lamda == 0):
        return softIn

    x = torch.zeros(softIn.shape).to(device)


    k = softIn > th
    x[k] = softIn[k] - th

    k = torch.abs(softIn) <= th
    x[k] = 0

    k = softIn < -th
    x[k] = softIn[k] + th
    
    #x = x.to(device)

    return x



# initially: N = 0 /  X_True = imgs # input / X_N = X_True - N
def fit(Encoder_model, Decoder_model, X_True, X_N, optimizer, criterion): 

    optimizer.zero_grad()

    # tuning parameters
    lamda = 10.0 #0.75 # 1.0 # 10.0 # 100.0
    mu = 0.02

    enc_out = Encoder_model(X_N)
    output = Decoder_model(enc_out)

    loss_1 = criterion(output, X_True) # MSE

    softThresholdIn = X_True - output

    #start_time = time.process_time()
    # learn N (noise or foreground) using soft-thresholding technique
    N = soft_threshold(lamda, softThresholdIn.detach())
    loss_2 = torch.mean(torch.abs(N)) # l1-norm
    #print("Processing time of computing N = ", time.process_time()-start_time, " s")

    loss_3 = 0
    for name, param in Encoder_model.named_parameters():
        loss_3 += torch.mean(param.data * param.data) # L2-norm

    for name, param in Decoder_model.named_parameters():
        loss_3 += torch.mean(param.data * param.data) # L2-norm
    
    #loss_2.detach_()
    #loss_3.detach_()

    loss = loss_1 + lamda * loss_2.item() + (mu/2.0) * loss_3.item()



    loss.backward()

    optimizer.step()

    loss_1.detach_()
    loss_2.detach_()
    loss_3.detach_()
    loss.detach_()


    del output 

    return loss, N, loss_1, loss_2, loss_3



def trainRCAE(dataloader, print_epoch=1, verbose=True):

    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if image_size_W == image_size_H:
        image_size = image_size_H


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
    
    if initialize_weights:
            Encoder_model.apply(weights_init)
            Decoder_model.apply(weights_init)

    if loss_ == True:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()  

    optimizer = torch.optim.Adam(list(Encoder_model.parameters())+list(Decoder_model.parameters()), lr=lr)

    print("Starting Training Loop...")

    N = torch.zeros(batch_size, nc, image_size, image_size).to(device)
    AE_losses = []
    img_list = []
    img_list_N = []
    # For each epoch
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        Encoder_model.train()
        Decoder_model.train()
        gc.collect()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            """
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        del obj
                except:
                    pass
            """
            if verbose: print(data[0].shape)
            if verbose: print(data[1].shape)

            X_True = data[0].to(device)
            X_N = X_True - N

            #start_time = time.process_time()
            recons_loss, N_tmp, loss_1, loss_2, loss_3 = fit(Encoder_model, Decoder_model, X_True, 
                                                                      X_N, optimizer, criterion)
            N = N_tmp
            if verbose: print("noise shape: ", N_tmp.shape)
            #print("Processing time of fit = ", time.process_time()-start_time, " s")

            # Output training stats
            if i % print_epoch == 0:
                print('Loss_AE: %.4f\tloss_N: %.4f\tloss_w: %.4f'
                        % (loss_1.item(), loss_2.item(), loss_3.item()))
                print('[%d/%d][%d/%d]\tLoss_AE: %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader),
                           recons_loss.item()))

            # Save Losses for plotting later
            AE_losses.append(recons_loss.item())
           

            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    Encoder_model.eval()
                    Decoder_model.eval()

                    enc_out = Encoder_model(data[0].to(device))
                    img_out = Decoder_model(enc_out).detach().cpu()

                img_list.append(vutils.make_grid(img_out[0:10], nrow=5, normalize=True))
                img_list_N.append(vutils.make_grid(N.cpu()[0:10], nrow=5, normalize=True))

    return AE_losses, img_list, img_list_N, Encoder_model, Decoder_model




