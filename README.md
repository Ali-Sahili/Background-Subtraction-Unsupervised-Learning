# Background-Subtraction-Unsupervised-Learning
Background Subtraction for complex scenes such as intersections from surveillance cameras

## Introduction
In the last few years, deep learning based generative models have gained more and more interest due to (and implying) some amazing improvements in the field. Relying on huge amount of data, well-designed networks architectures and smart training techniques, deep generative models have shown an incredible ability to produce highly realistic pieces of content of various kind, such as images, texts and sounds. Among these deep generative models, two major families stand out and deserve a special attention: Generative Adversarial Networks (GANs) and Autoencoders (AEs).

## Project’s parts
In this project, I implemented a deep convolutif GAN model along with three different families of Auto-encoders: traditional auto-encoder(AE), variational auto-encoder(VAE) and adversarial auto-encoder(AAE).

### DCGANs
Deep Convolutional Generative Adversarial Networks (DCGANs for short) have had a huge success. It has same methodlogy of traditional GANs which are algorithmic architectures that use two neural networks, pitting one against the other (thus the adversarial) in order to generate new, synthetic instances of data that can pass for real data. In this project, using DCGANs to reconstruct the background do not lead to convergence or an acceptable result.

### AutoEncoders
The general idea of autoencoders is pretty simple and consists in setting an encoder and a decoder as neural networks and to learn the best encoding- decoding scheme using an iterative optimisation process. So, at each iteration we feed the autoencoder architecture (the encoder followed by the decoder) with
1some data, we compare the encoded-decoded output with the initial data and backpropagate the error through the architecture to update the weights of the
networks. Thus, intuitively, the overall autoencoder architecture (encoder+decoder) creates a bottleneck for data that ensures only the main structured part
of the information can go through and be reconstructed. Looking at our general framework, the family E of considered encoders is defined by the encoder network architecture(for different input image size like 64, 128, 256 and 512), the family D of considered decoders is defined by the decoder network architecture
and the search of encoder and decoder that minimise the reconstruction error is done by gradient descent over the parameters of these networks.
In this part, we tried 4 main types: DCGAN, AE, AAE and VAE. Trying different image sizes and several sizes for the latent space, we found a good result for image size of 512x512x3 and a latent space of size 1x1. Also, we use a combinaison of two loss functions (MSE and similarity loss) which led to a better result but in general the stopping vehhicules at the traffic light still considered as a part of the background. In addition, changing between the three types of auto-encoders did not affect significantly the results, so we focus precisly on the traditional autoencoders.

## Organization
For all of these codes (DCGAN, AE, AAE, and VAE), we have mainly six files:
#### 1. main.py the main file
#### 2. model.py 
this code explain the network architecture of the model as classes
#### 3. Param.py 
this file contains all parameters of the model like learning rate, number of epochs, paths of input and output datas, batch size, input image size, number of GPUs, the size of the latent space and so on. Prepare data.py this code read the data and put as a torch dataset which helps to shuffle, normalize and divid it according to the expected batch size .
#### 4. utils.py 
some useful functions for plotting losses and results from pytorch librairies
#### 5. train.py training file

### requirements
• python3
• pytorch
• torchvision
• numpy
• matplotlib
In case of using the similarity loss, we must install its library: pytorch ssim. To install this library, put into your terminal:
git clone https://github.com/Po-Hsun-Su/pytorch-ssim
python3 setup.py install

## Usage
To generate your own model, put the following one in your terminal and choose your desired architecture: python3 main.py
You can change the parameters of the model and the paths of the input and the output data in the Param.py file.
In this project, several architecture and methods are implemented based mainly on the previous ones:
• Convolutional Auto-encoder (AE)
• Variational Auto-Encoder (VAE)
• Adverserial Auto-Encoder (AAE)
• Couple of two auto-encoders (CAE)
• Combination between optical flow and RGB image at input or with a binary mask obtained by a simple BG method.
• Deep convolutional GAN with a modified loss function (DCGAN)
• Denoising Auto-Encoder (DAE)
• Patch-level Auto-Encoder
• Patch-level Auto-Encoder with attention modules
• one encoder with several decoders at different encoder’s parts
• multi-scale inputs (rgb image HxW, H/2xW/2, H/4xW/4)
• Two encoders: one is for RGB images and the second is for their optical flow map and One decoder to decode the fused information at the latent space.

We can switch between all these models by changing their corresponding parameters in the Param.py file.
Also, two background initialization methods are presented, the first is simple and based on an average filter along the batch size while the second used a convolutional layer with average and max pooling layers to achieve the same idea. Their models are written in the background initialzation.py file.

## Our Test
In the last few years, deep learning based generative models have gained more and more interest due to (and implying) some amazing improvements in the field. Relying on huge amount of data, well-designed networks architectures and smart training techniques, deep generative models have shown an incredible ability to produce highly realistic pieces of content of various kind, such as images, texts and sounds. Among these deep generative models, two major families stand out and deserve a special attention: Generative Adversarial Networks (GANs) and Auto-encoders (AEs). As a consequence, we focus mainly, in this project, on these two families with some extensions. A brief explanation of all constructed models and their architectures with their results are presented below.
