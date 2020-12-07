import torch



# Root directory for dataset
#dataroot = '../../../input/Sequence/'
dataroot = '../../../Varna_datasets/Varna_36000/'
#dataroot ='/home/simon/Desktop/Scene_Analysis/CDnet/dataset2014/dataset/shadow/bungalows/input/'

#dataroot2 = '../../../Varna_datasets/FlowsImgs/'
dataroot2 = '../../../Varna_datasets/FlowsImgs/'



# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size_H = 256 #720
image_size_W = 256 #1280

if image_size_H <= image_size_W:
    image_size = image_size_H
else:
    image_size = image_size_W

# Number of channels in the training images. For color images this is 3
nc = 3
nc_2 = 3 # number of channels of optical flow map or mask

# Size of z latent vector (i.e. size of generator/decoder input)
nz = 50

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64
nf = 8

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.0001 # 0.0002

# Learning rate for optimizers for AAE
gen_lr = 0.0001
reg_lr = 0.00005
TINY = 1e-15

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
Tensor = torch.cuda.FloatTensor if (torch.cuda.is_available() and ngpu > 0) else torch.FloatTensor

#method = 'patchLevel_withoutAttention' # apply AE concept on patches extracted from images
#method = 'patchLevel_withAttention'  
#method = 'Couple_Autoencoders'      # two cascaded auto-encoders
#method = 'Autoencoder'              # convolutional auto-encoder
#method = 'AE_Attention'# convolutional AE with attention modules applied on optical flow map
#method = 'atrous_AE'
#method = 'atrous_AE_fuse'     
#method = 'atrous_AE_new'      # AE with ASPP block between encoder and decoder
#method = 'trainAE_Gray'       # add a gray image to the input (4-channel input)
#method = 'AE_swish'          # replace ReLU by Swish function in all layers
#method = 'AE_Unet'         # AE with a Unet architecture           
#method = 'AE_GroupNorm'  # replace batch norm with group norm in decoder and encoder layers
method = 'AE_deformConv'
#method = 'MultiScale_input'         # input is multi-scaled to 512/256/128
#method = 'OneEncoder_MultiDecoders' # using decoder at different levels of the encoder's layers
#method = 'RCAE'                      # Robust Convolutional auto-encoder
#method = 'AE_GMMN'                  # AE with generative moments matching network
#method = 'WAE_GAN'
#method = 'WAE_MMD'
#method = 'VSC'                      # variational sparse coding
#method = 'VAE'                      # variational auto-encoder
#method = 'VAE_Attention'
#method = 'OneEncoder_MultiDecoders_VAE' #  decoding at different levels
#method = 'MultiScale_VAE'
#method = 'NVAE'
#method = 'AAE'                       # adversarial auto-encoder
#method = 'WAAE'                       # wasserstein adversarial auto-encoder
#method = 'DCGAN'                    # Deep convolutional GAN
#method = 'WDCGAN'                    # Wasserstein Deep convolutional GAN
#method = 'modified_DCGAN'# modifying the traditional DCGAN with a new loss function and U-net
#method = 'OneEncoders_OneDecoder'   # using optical flow
#method = 'TwoEncoders_OneDecoder'   # using optical flow

#method = 'AE_pretrained_model'
#method = 'ConvLSTM'                 
#method = 'PICA'

#method = 'AE_Attention_Encoder'
ATTENTION_M =  'CBAM_Module'  # 'Spatio_ChannelWise' #  

#method = 'complex_attention_model'

patch_mode = False

# Initialze the weights of encoder and decoder
initialize_weights = True

# Saving the output model
save_outputModel = True

# Using the concept of denoising autoencoder instead of simple autoencoder
denoising_autoencoder = False #denoising autoencoders

# Initialize background at the beginning
Initializze_BG = False

# loss_= true --> use the binary cross entropy as loss function, otherwise, mean square error
loss_ = False

# for DCGAN method, use Unet architecture for the discriminator network
Unet = False


Allow_Shuffle = False

#SN_ind = True # use spectral normalization instead of batch normalization in the discriminator of dcgan








