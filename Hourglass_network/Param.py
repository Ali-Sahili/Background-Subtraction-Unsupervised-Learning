import torch



# Root directory for dataset
#dataroot = '../../../Varna_datasets/Varna_36000/'

dataroot = 'Imgs/'
dataroot_mask = 'Masks/'


# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256 

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator/decoder input)
nz = 50

# parameters of hourglass network
nstack = 1
oup_dim = 16
bn = False
increase = 0

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0001 # 0.0002

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
Tensor = torch.cuda.FloatTensor if (torch.cuda.is_available() and ngpu > 0) else torch.FloatTensor



# Initialze the weights of encoder and decoder
initialize_weights = True

# Saving the output model
save_outputModel = True


Allow_Shuffle = False









