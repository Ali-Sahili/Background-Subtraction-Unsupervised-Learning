import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.utils as vutils
import torchvision.transforms as transforms

import gc

from VAE.model import *

import os
path = os.getcwd()
os.chdir("../")
from Param import *
os.chdir(path)


from VAE.Loss_functions import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_tensor(input, size):
        final_output = None
        batch_size, channels, h, w = input.shape
        input_ = torch.squeeze(input, 1)

        for img in input_:
            img_PIL = transforms.ToPILImage()(img)
            img_PIL = transforms.Resize((size,size))(img_PIL)
            img_PIL = transforms.ToTensor()(img_PIL)
            if final_output is None:
                final_output = img_PIL
            else:
                final_output = torch.cat((final_output, img_PIL), 0)

        final_output = torch.unsqueeze(final_output, 1)
        return final_output.view(batch_size, channels, size, size)


def trainMSVAE(dataloader, print_epoch = 32, multi_=3):

    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    ## scale 512x512
    model512 = VAE()
    optimizer512 = torch.optim.Adam(model512.parameters(), lr=lr) 
    model512 = model512.to(device)

    ## scale 256x256
    model256 = VAE256()
    optimizer256 = torch.optim.Adam(model256.parameters(), lr=lr) 
    model256 = model256.to(device)

    ##   scale 128x128
    model128 = VAE128()
    optimizer128 = torch.optim.Adam(model128.parameters(), lr=lr) 
    model128 = model128.to(device)

    img_list = []
    losses = []

    print("Starting Training Loop...")
    

    for epoch in range(num_epochs):
        gc.collect()

        model512.train()
        model256.train()
        model128.train()

        for i, data in enumerate(dataloader, 0):
            torch.cuda.empty_cache()
            img512 = Variable(data[0]).to(device)

            #   scale 512
            optimizer512.zero_grad()
            out512, mu512, logvar512 = model512(img512)

            #    scale 256
            optimizer256.zero_grad()
            img256 = resize_tensor(img512.cpu(), size=256).detach().to(device)
            out256, mu256, logvar256 = model256(img256)
            out256_512 = F.interpolate(out256, scale_factor=2)

            #    scale 128
            optimizer128.zero_grad()
            img128 = resize_tensor(img512.cpu(), size=128).detach().to(device)
            out128, mu128, logvar128 = model128(img128)
            out128_512 = F.interpolate(out128, scale_factor=4)
   
            out_avg = (out512 + out256_512 + out128_512)/3.0
            del out256_512, out128_512, out128, out256
  
   
            out_avg256 = resize_tensor(out_avg.cpu(), size=256).to(device)
            out_avg128 = resize_tensor(out_avg.cpu(), size=128).to(device)

            mu = (mu128 + mu256 + mu512)/3.0
            logvar = (logvar128 + logvar256 + logvar512)/3.0
            del mu128, mu256, mu512, logvar128, logvar256, logvar512 


            loss512 = loss_function2(out512, img512, mu, logvar)
            loss512.backward(retain_graph=True)
            optimizer512.step()
            loss512.detach_()
            del out512


            loss256 = loss_function2(out_avg256, img256, mu, logvar)
            loss256.backward(retain_graph=True)
            optimizer256.step()
            loss256.detach_()
            del out_avg256


            loss128 = loss_function2(out_avg128, img128, mu, logvar)
            loss128.backward()
            optimizer128.step()
            loss128.detach_()
            del out_avg128

            del mu, logvar


            losses.append(loss512)


            if i % print_epoch == 0:
                    print('[%d/%d][%d/%d]\tLoss: %.4f  %.4f  %.4f'
                        % (epoch+1, num_epochs, i, len(dataloader), loss128.item(), loss256.item(), 
                                                                                 loss512.item()))
                    del loss128, loss256, loss512 
                

            # Check how the generator is doing by saving G's output on fixed_noise
            if (epoch == num_epochs-1) and (i == len(dataloader)-1):
                img_list.append(vutils.make_grid(out_avg.detach().cpu(), padding=2, normalize=True))

            del out_avg
            gc.collect()

    return img_list, losses, model512
