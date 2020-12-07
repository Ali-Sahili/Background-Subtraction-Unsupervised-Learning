import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models

import cv2
import numpy as np
import skimage.io as io
import os

from Prepare_Data import DataLoader
from Param import *
from utils import *

#from Atrous_Convolution.Encoders import Encoder256_Atrous_fuse, Encoder256_Atrous
#from Atrous_Convolution.Decoders import Decoder256
#from Atrous_Convolution.model import Model512
from AE_swish.Encoders import Encoder512_swish
from AE_swish.Decoders import Decoder512_swish
#from models.Encoders import Encoder256, Encoder512, Encoder512_G
#from models.Decoders import Decoder256, Decoder512, Decoder512_G
from models.patchmodels import Encoder512_F, Decoder512_F
from models.patchmodels_Attention import Encoder_Decoder512
from models.AE_Attention import AE_Attention
from OneEncoder_MultiDecoders.models import VAE_Model512 , Model512
from MultiScale_input.model import FinalModel
from VAE.VAE_Attention import VAE_Attention_model
from VAE.model import VAE
from NVAE.vae_celeba import NVAE
from NVAE.utils import add_sn
#from AAE.models import Encoder512, Decoder512, Encoder256, Decoder256
from Unet.models import Encoder_U, Decoder_U
from VAE.model import Encoder, Decoder, VAE256
from VSC.model import VSC_Encoder, VSC_Decoder, VSC256
from AE_GMMN.GMMN import GMMN
from DCGAN.models import Generator512
from Attention_On_Encoder.AE_Attention_Encoder import AE_Attention_Encoder
#1from AE_pretrained_models.model import Encoder256, Decoder256
from AE.Complex_Attention import * 
from DeformConv.model import DeformConvNet_Encoder256, Decoder256


def toGray(input):
        final_output = None
        batch_size, channels, h, w = input.shape
        input_ = torch.squeeze(input, 1)

        for img in input_:
            img_PIL = transforms.ToPILImage()(img.cpu())
            img_PIL = img_PIL.convert('L')
            img_PIL = transforms.ToTensor()(img_PIL)
            #print(img_PIL.shape)
            if final_output is None:
                final_output = img_PIL
            else:
                final_output = torch.cat((final_output, img_PIL), 0)

        final_output = torch.unsqueeze(final_output, 1)
        return final_output.view(batch_size, 1, h, w).to(device)




# Create the dataloader
dataloader = DataLoader(dataroot, 256, 256, 1, 2, Allow_Shuffle=False)
dataloader2 = DataLoader(dataroot2, 256, 256, 1, 2, Allow_Shuffle=False)

torch.cuda.empty_cache()

# Create encoder and decoder models
"""
model = VSC256().to(device)
model.load_state_dict(torch.load('output_CDnet2014/VSC_256_batch32_epochs100_nz50_streetCornerAtNight.pth'))
model.eval()
"""


Encoder = DeformConvNet_Encoder256().to(device) #.apply(add_sn)
Decoder = Decoder256().to(device) #.apply(add_sn)
Encoder.load_state_dict(torch.load('output_models/AE_deformConv_Encoder256_batch32_epochs50_nz50.pth'))
Decoder.load_state_dict(torch.load('output_models/AE_deformConv_Decoder256_batch32_epochs50_nz50.pth'))
Encoder.eval()
Decoder.eval()


"""
gmmn_model = GMMN(50,50).to(device) #.apply(add_sn)
gmmn_model.load_state_dict(torch.load('output_CDnet2014/AE_GMMN_GMMN256_batch32_epochs100_nz50_streetCornerAtNight.pth'))
gmmn_model.eval()
"""

method = 'AE_deformConv_256_batch32_epochs50_nz50'
out_path = 'output_varna/' + method
if not os.path.exists(out_path):
    os.makedirs(out_path)

  
N = 0
real_img_list = []
img_list = []

print('Starting testing ...')
for i, (data1,data2) in enumerate(zip(dataloader, dataloader2), 0):
    print(i)
    
    if i%1000==0 or i>30000:
        #imgs = data[0].to(device)
        #print(imgs.shape)
        img1 = data1[0].to(device)
        img2 = data2[0].to(device) 

        imgs = torch.cat((img1,img2),1)       

        #output,_,_,_ = model(img1)

        #input_ = img1 - N
        #gray = toGray(img1)
        #in_imgs = torch.cat((img1,gray),1).to(device)

        enc_out = Encoder(img1)
        #enc_out = gmmn_model(enc_out1)
        output = Decoder(enc_out)


        #lamda = 1.0
        #softIn = img1 - output
       # N = soft_threshold(lamda, softIn)

        input_data = img1.detach().cpu()
        output = output.detach().cpu()[0]
        #N_img = N.detach().cpu()
        #print(output.shape)

        in_img = vutils.make_grid(input_data, padding=2, normalize=True).permute(1,2,0).numpy()
        bg_img = vutils.make_grid(output, padding=2, normalize=True).permute(1,2,0).numpy()
        #N_img = vutils.make_grid(N_img, padding=2, normalize=True).permute(1,2,0).numpy()

        #print('input: ', in_img.shape, in_img.dtype)
        #print('output: ', bg_img.shape, bg_img.dtype)

        #fg_img = np.where(in_img>bg_img, in_img-bg_img, 0)
        fg_img = cv2.subtract(bg_img, in_img)
        

        #cv2.imshow('input', in_img)
        #cv2.imshow('background', bg_img)
        #cv2.imshow('foreground', fg_img)
        #cv2.waitKey(0)

        io.imsave(out_path + '/' + str(i) +'_in.jpg', in_img)
        io.imsave(out_path + '/' + str(i) +'_bg.jpg', bg_img)
        io.imsave(out_path + '/' + str(i) +'_fg.jpg', fg_img)
        #io.imsave(out_path + '/' + str(i) +'_fg2.jpg', fg_img2)


