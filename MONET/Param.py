import torch



# Root directory for dataset
dataroot = '../../../Varna_datasets/Varna_36000/'


# MONET model parameters
image_size = 128
nc = 3
nf = 64
n_heads = 6
n_slots = 5
n_iters = 3

if image_size == 128:
    n_stack = 4
elif image_size == 256:
    n_stack = 5
elif image_size == 512:
    n_stack = 6
else:
    raise NotImplementedError

# Learning rate for optimizers
lr = 0.0001 # 0.0002

# Batch size during training
batch_size = 32



# Number of training epochs
num_epochs = 1

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Number of workers for dataloader
workers = 4 * ngpu

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
Tensor = torch.cuda.FloatTensor if (torch.cuda.is_available() and ngpu > 0) else torch.FloatTensor



# Initialze the weights of encoder and decoder
initialize_weights = True

# Saving the output model
save_outputModel = True


Allow_Shuffle = False














