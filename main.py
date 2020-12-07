from Prepare_Data import DataLoader


from Param import *
from utils import *

import time


print()
# Create the dataloader
dataloader = DataLoader(dataroot, image_size_H, image_size_W, batch_size, workers, Allow_Shuffle=Allow_Shuffle)
print('Number of images used in training: ', len(dataloader)*batch_size)

# Plot some training images
#Data_plot(dataloader, image_size)

print()
print('Used model: ', method)
print('Input image size: ', image_size)
print('Latent space dimension: ', nz)
print('Batch size: ', batch_size)
print('Number of epochs: ', num_epochs)
print()

start_time = time.process_time()


###################################
##      Couple of two auto-encoder
###################################
if method == 'Couple_Autoencoders':
    from CoupleAE.train import trainCAE

    AE_losses_1, AE_losses_2, img_list1, img_list2, Encoder1, Encoder2, Decoder1, Decoder2 = trainCAE(dataloader, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    
    if save_outputModel:
        encoder_name = method+"_Encoder1_"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_bungalows.pth"
        decoder_name = method+"_Decoder1_"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_bungalows.pth"
        save_model(Encoder1, encoder_name)
        save_model(Decoder1, decoder_name)

        encoder_name = method+"_Encoder2_"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_bungalows.pth"
        decoder_name = method+"_Decoder2_"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_bungalows.pth"
        save_model(Encoder2, encoder_name)
        save_model(Decoder2, decoder_name)
    

    AE_losses_plot(AE_losses_1)
    AE_losses_plot(AE_losses_2)

    Results_plot(dataloader, img_list1)
    Results_plot(dataloader, img_list2)


###################################
##      Variational auto-encoder
###################################
elif method == 'VAE':
    from VAE.train import trainVAE

    img_list, losses, model, mu_list, logvar_list = trainVAE(dataloader)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        model_name = method+ "_"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_streetCornerAtNight.pth"

        save_model(model, model_name)

    
    AE_losses_plot(losses)

    Results_plot(dataloader, img_list)



###################################
##      Variational Sparse Coding
###################################
elif method == 'VSC':
    from VSC.train import trainVSC

    img_list, losses, recon_losses, model = trainVSC(dataloader)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        model_name = method+ "_"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_streetCornerAtNight.pth"
        save_model(model, model_name)

    
    AE_losses_plot(losses)
    AE_losses_plot(recon_losses)

    Results_plot(dataloader, img_list)


##############################################
##      Multi-scale Variational auto-encoder
##############################################
elif method == 'MultiScale_VAE':
    from VAE.train3 import trainMSVAE

    img_list, losses, model = trainMSVAE(dataloader)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:

            model_name = method +str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
            save_model(model, model_name)
    
    AE_losses_plot(losses)

    Results_plot(dataloader, img_list)



###################################
##      Adverserial Auto-encoder
###################################
elif method == 'AAE' or method == 'WAAE':
    from AAE.train import trainAAE 

    AAE_losses, G_losses, D_losses, img_list, Encoder, Decoder = trainAAE(dataloader, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:

            Enc_model_name = method +'_Encoder' +str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_MSE_SSIM_streetCornerAtNight.pth"
            Dec_model_name = method +'_Decoder'+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_MSE_SSIM_streetCornerAtNight.pth"
            save_model(Encoder, Enc_model_name)
            save_model(Decoder, Dec_model_name)

    AE_losses_plot(AAE_losses, name="reconstruction loss")
    AE_losses_plot(G_losses, name="generator loss")
    AE_losses_plot(D_losses, name="discriminator loss")

    Results_plot(dataloader, img_list)


###################################
##  Deep convolutional GAN architecture
###################################
elif method == 'DCGAN':
    from DCGAN.train import trainDCGAN

    G_losses, D_losses, img_list, model = trainDCGAN(dataloader, print_epoch=32, verbose=False)
    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        if SN_ind:
            genertor_name = method + "_Generator_SN_"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        else:
            genertor_name = method + "_Generator"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(model, genertor_name)


    G_D_losses_plot(G_losses, D_losses)
    Results_plot(dataloader, img_list)


###################################
##  Deep convolutional GAN architecture
###################################
elif method == 'WDCGAN':
    from WDCGAN.train import trainWDCGAN

    G_losses, D_losses, img_list, model = trainWDCGAN(dataloader, print_epoch=32, verbose=False)
    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        genertor_name = method + "_Generator"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(model, genertor_name)


    G_D_losses_plot(G_losses, D_losses)
    Results_plot(dataloader, img_list)


############################################################################################
##  Deep convolutional GAN architecture with Unet architecture and a modified loss function
############################################################################################
elif method == 'modified_DCGAN':
    from modified_DCGAN.train import trainDCGAN2

    G_losses, D_losses, img_list, model = trainDCGAN2(dataloader, print_epoch=32, verbose=False)
    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        genertor_name = method + "_Generator"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(model, genertor_name)

    G_D_losses_plot(G_losses, D_losses)
    Results_plot(dataloader, img_list)


###############################################################################
##   One encoder model with several decoder at different level of the encoder
###############################################################################
elif method == 'OneEncoder_MultiDecoders' or method == 'OneEncoder_MultiDecoders_VAE':
    from OneEncoder_MultiDecoders.train import trainOEMD

    AE_losses, img_list, model = trainOEMD(dataloader, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        AE_name = "OEMD"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(model, AE_name)

    AE_losses_plot(AE_losses, name="reconstruction loss")

    Results_plot(dataloader, img_list)


###############################################################################
##  Apply attention modules on encoder part of auto-encoder
###############################################################################
elif method == 'AE_Attention_Encoder':
    from Attention_On_Encoder.train import trainAOE

    AE_losses, img_list, model = trainAOE(dataloader, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        AE_name = method+"_" + ATTENTION_M +str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(model, AE_name)

    AE_losses_plot(AE_losses, name="reconstruction loss")

    Results_plot(dataloader, img_list)


################################################################################
##   two input have been fused nc =6 using rgb image and its optical flow map
################################################################################
elif method == 'OneEncoders_OneDecoder':
    from Combination.train import train1E1D

    dataloader2 = DataLoader(dataroot2, image_size_H, image_size_W, batch_size, workers, Allow_Shuffle=False)

    AE_losses, img_list, Encoder, Decoder = train1E1D(dataloader,dataloader2, print_epoch=32, verbose=False)

    if save_outputModel:
            encoder_name = method + "Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
            decoder_name = method + "Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
            save_model(Encoder, encoder_name)
            save_model(Decoder, decoder_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)



############################################################################################
##      two encoders: one for RGB image and the other for its optical flow map/ one decoder
############################################################################################
elif method == 'TwoEncoders_OneDecoder':
    from Combination.train2 import train2E1D

    dataloader2 = DataLoader(dataroot2, image_size_H, image_size_W, batch_size, workers, 	                                   Allow_Shuffle=False)
    AE_losses, img_list, Encoder, Encoder2, Decoder = train2E1D(dataloader,dataloader2, 
                                                                 print_epoch=32, verbose=False)
    ###
    if save_outputModel:
        encoder_name = method+"Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        decoder_name = method+"Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(Encoder, encoder_name)
        save_model(Decoder, decoder_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)



###################################
##           ConvLSTM auto-encoder
###################################
elif method == 'ConvLSTM':
    from ConvLSTM.train import trainConvLSTM

    losses, Encoder, Decoder = trainConvLSTM(dataloader, print_epoch=32)
    print("\nProcessing time = ", time.process_time()-start_time, " s")
    ###
    if save_outputModel:
        encoder_name = method+"Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        decoder_name = method+"Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(Encoder, encoder_name)
        save_model(Decoder, decoder_name)
    
    AE_losses_plot(losses)


###################################
##         PICANet architecture
###################################
elif method == 'PICA':
    from PICANet.train import trainPICA
 
    losses, img_list, model = trainPICA(dataloader, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        AE_name = "PICA"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(model, AE_name)

    AE_losses_plot(losses, name="reconstruction loss")

    Results_plot(dataloader, img_list)

#################################################################



###################################
##      AE with GMMN
###################################
elif method == 'AE_GMMN':
    from AE_GMMN.train import trainAE_GMMN

    losses, img_list, gmmn_model, AE_Encoder, AE_Decoder = trainAE_GMMN(dataloader)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        model_name = method+ "_GMMN"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_streetCornerAtNight.pth"
        save_model(gmmn_model, model_name)

        encoder_name = method+"Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_streetCornerAtNight.pth"
        decoder_name = method+"Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_streetCornerAtNight.pth"
        save_model(AE_Encoder, encoder_name)
        save_model(AE_Decoder, decoder_name)

    
    AE_losses_plot(losses)

    Results_plot(dataloader, img_list)

######################################################################
##        Autoencoder with attention modules architecture
######################################################################
elif method == 'patchLevel_withAttention' or method == 'AE_Attention': 
    from AE.train2 import trainPWA

    dataloader2 = DataLoader(dataroot2, image_size_H, image_size_W, batch_size, workers, 	                                   Allow_Shuffle=False)
    AE_losses, img_list, AE_model = trainPWA(dataloader, dataloader2, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        AE_name = method +str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"-attendToBG.pth"
        save_model(AE_model, AE_name)

    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)
#####################################################################

######################################################################
##    Variational Autoencoder with attention modules architecture
######################################################################
elif method == 'VAE_Attention': 
    from VAE.train2 import trainVAEWA

    dataloader2 = DataLoader(dataroot2, image_size_H, image_size_W, batch_size, workers, 	                                   Allow_Shuffle=False)
    VAE_losses,img_list, VAE_model = trainVAEWA(dataloader, dataloader2, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        VAE_name = method +str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"-attendToBG.pth"
        save_model(VAE_model, VAE_name)

    
    AE_losses_plot(VAE_losses)

    Results_plot(dataloader, img_list)
#####################################################################

######################################################################
##    Unet architecture
######################################################################
elif method == 'AE_Unet':
    from Unet.train import trainUNet

    AE_losses, img_list, Encoder, Decoder = trainUNet(dataloader, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")


    if save_outputModel:
            encoder_name = method +"_Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
            decoder_name = method +"_Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
            save_model(Encoder, encoder_name)
            save_model(Decoder, decoder_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)
#####################################################################


######################################################################
##    2D Convultion with 2D dilated concolution in encoder layers
######################################################################
elif method == 'atrous_AE' or method == 'atrous_AE_fuse':
    from Atrous_Convolution.train import train_atrousAE

    AE_losses, img_list, Encoder, Decoder = train_atrousAE(dataloader, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")


    if save_outputModel:
            encoder_name = method +"_Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_streetCornerAtNight.pth"
            decoder_name = method +"_Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_streetCornerAtNight.pth"
            save_model(Encoder, encoder_name)
            save_model(Decoder, decoder_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)
#####################################################################

######################################################################
##    2D Convultion with 2D dilated concolution in encoder layers
######################################################################
elif method == 'atrous_AE_new':
    from Atrous_Convolution.train2 import train_atrousAE_new

    AE_losses, img_list, model = train_atrousAE_new(dataloader, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")


    if save_outputModel:
            model_name = method +"_"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
            save_model(model, model_name)

    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)
#####################################################################


######################################################################
##    Auto-encoder with swish activation function
######################################################################
elif method == 'AE_swish':
    from AE_swish.train import trainAE_swish

    AE_losses, img_list, Encoder, Decoder = trainAE_swish(dataloader, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")


    if save_outputModel:
            encoder_name = method +"_Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
            decoder_name = method +"_Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
            save_model(Encoder, encoder_name)
            save_model(Decoder, decoder_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)
#####################################################################


######################################################################
##    Nouveau VAE
######################################################################
elif method == 'NVAE':
    from NVAE.train import train_NVAE

    losses, img_list, model = train_NVAE(dataloader, print_epoch=32)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
            model_name = method + str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"

            save_model(model, model_name)
    
    AE_losses_plot(losses)

    Results_plot(dataloader, img_list)
#####################################################################

# add gray image to the input --> 4 channels input
elif method == 'trainAE_Gray':
    from AE.train3 import trainAE_Gray        
    AE_losses, img_list, Encoder, Decoder = trainAE_Gray(dataloader, print_epoch=32, verbose=False)

 
    if save_outputModel:
        encoder_name = method +"_Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        decoder_name = method +"_Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(Encoder, encoder_name)
        save_model(Decoder, decoder_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)
###############################################################################333

elif method == 'RCAE':
    from RCAE.train import trainRCAE        
    AE_losses, img_list, img_list_N, Encoder, Decoder = trainRCAE(dataloader, print_epoch=32, verbose=False)
    print("\nProcessing time = ", time.process_time()-start_time, " s")
 
    if save_outputModel:
        encoder_name = method +"_Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        decoder_name = method +"_Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(Encoder, encoder_name)
        save_model(Decoder, decoder_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)
    Results_plot(dataloader, img_list_N)

###############################################################################333

################################################################################
##   extract features using a pretrained model inside encoder part
################################################################################
elif method == 'AE_pretrained_model':
    from AE_pretrained_models.train import trainAE

    AE_losses, img_list, Encoder, Decoder = trainAE(dataloader, print_epoch=32, verbose=False)

    if save_outputModel:
            encoder_name = method + "Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_alexnet_streetCornerAtNight.pth"
            decoder_name = method + "Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_alexnet_streetCornerAtNight.pth"
            save_model(Encoder, encoder_name)
            save_model(Decoder, decoder_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)

###############################################################################333
##
elif method == 'WAE_GAN' or method == 'WAE_MMD':
    if method == 'WAE_GAN':
        from WAE.WAE_GAN import trainWAE_GAN    
        AE_losses,img_list, Encoder,Decoder = trainWAE_GAN(dataloader,print_epoch=32, verbose=False)
    elif method == 'WAE_MMD':
        from WAE.WAE_MMD import trainWAE_MMD    
        AE_losses,img_list, Encoder,Decoder = trainWAE_MMD(dataloader,print_epoch=32, verbose=False)
 
    if save_outputModel:
        encoder_name = method +"_Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_streetCornerAtNight.pth"
        decoder_name = method +"_Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_streetCornerAtNight.pth"
        save_model(Encoder, encoder_name)
        save_model(Decoder, decoder_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)
###############################################################################333

elif method == 'complex_attention_model':   
    from AE.train_complex import train

    AE_losses, img_list, att_enc_list, att_dec_list, model = train(dataloader)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    if save_outputModel:
        model_name = method +"_"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(model, model_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)
    Results_plot(dataloader, att_enc_list)
    Results_plot(dataloader, att_dec_list)


####################################################################################

elif method == "AE_deformConv":
    from DeformConv.train import trainAE

    AE_losses, img_list, Encoder, Decoder = trainAE(dataloader, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")
    
    if save_outputModel:
        encoder_name = method +"_Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        decoder_name = method +"_Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
        save_model(Encoder, encoder_name)
        save_model(Decoder, decoder_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)
#######################################################################################


else:
    from AE.train import trainAE

    if method == 'MultiScale_input':
        AE_losses, img_list, AE_model = trainAE(dataloader, print_epoch=32, verbose=False)
    else:
        AE_losses, img_list, Encoder, Decoder = trainAE(dataloader, print_epoch=32, verbose=False)

    print("\nProcessing time = ", time.process_time()-start_time, " s")

    
    
    if save_outputModel:
        if method == 'MultiScale_input':
            AE_name = method +str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+".pth"
            save_model(AE_model, AE_name)
        else:
            encoder_name = method +"_Encoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_PhotometricLoss_bungalows.pth"
            decoder_name = method +"_Decoder"+str(image_size)+"_batch"+str(batch_size)+"_epochs"+str(num_epochs)+"_nz"+str(nz)+"_PhotometricLoss_bungalows.pth"
            save_model(Encoder, encoder_name)
            save_model(Decoder, decoder_name)
    
    AE_losses_plot(AE_losses)

    Results_plot(dataloader, img_list)










