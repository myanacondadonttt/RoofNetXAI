import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

# takes a multi-dimensional input tensor and reshapes it into a two-dimensional tensor, 
# where the first dimension represents the batch size and the second dimension represents the flattened representation of the input
# first dimension represents the batch size and the second dimension represents the flattened
# representation of the input image
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



# takes a two-dimensional input tensor and reshapes it into a multi-dimensional tensor
# it reshapes the input tensor to have a size of (batch_size, 1024, 1, 1), effectively "unflattening" the input tensor into a higher-dimensional representation. 
# used in the decoder part of a neural network, where it converts a flattened latent space representation 
# into a spatially structured representation for generating images or reconstructing the input data
class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

# img_dim = spatial dimension of the input image
# image channels = number of channels in the input image (RGB images have 3 channels)
# h_dim = dimension of the hidden layer (the dimensionality of the intermediate representation in both the encoder and decoder) 
# z_dim = dimension of the latent space (the dimensionality of the latent space, or the dimensionality of the output of the encoder)
class VAE(nn.Module):
    def __init__(self, img_dim=32, image_channels=3, h_dim=1024, z_dim=128, device=None):
        super(VAE, self).__init__()

        assert img_dim == 32 or img_dim == 64 or img_dim == 255, 'img_dim must be 32 or 64 or 255'
        if img_dim == 32:
            self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=4, stride=2), #32 filters of size 4x4 with a stride of 2
                nn.ReLU(), # ReLU activation function element-wise to the output of the previous convolutional layer, introducing non-linearity
                nn.Conv2d(32, 64, kernel_size=4, stride=2), # the step size at which the kernel (filter) moves across the input image or feature map 
                #stride determines how much the kernel shifts or moves horizontally and vertically after each application of the kernel to the input image
                nn.ReLU(), 
                nn.Conv2d(64, 256, kernel_size=4, stride=2), #256 filtrs of size 4x4 with a stride of 2
                nn.ReLU(),
                Flatten() # flattens the output of the previous layer to a one-dimensional vector/tensor
            ).to(device)

            self.decoder = nn.Sequential(
                UnFlatten(), # reshapes the input from a flattened representation (shape: [batch_size, h_dim]) 
                # to a 4-dimensional tensor (shape: [batch_size, h_dim, 1, 1]
                nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),# increases the spatial dimensions 1x1 -> 5x5 
                nn.ReLU(), # sets negative values to zero and keeps non-negative values unchanged
                nn.ConvTranspose2d(128, 32, kernel_size=6, stride=2), #5x5 -> 14x14
                nn.ReLU(), 
                nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2), #14x14 -> 32x32
                nn.Sigmoid(), # squashes the output between the range of 0 and 1 (normilizing the pixel values)
                # ensures that the reconstructed image has pixel values in the valid range for the given image format 
                # (e.g., between 0 and 255 for grayscale or RGB images).
            ).to(device)

        elif img_dim == 64:
            self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=4, stride=2), #64x64 -> 31x31
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), #31x31 -> 14x14
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2), #14x14 -> 6x6
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2), #6x6 -> 2x2: 4x256=1024
                nn.ReLU(),
                Flatten()
            ).to(device)

            self.decoder = nn.Sequential(
                UnFlatten(),
                nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
                nn.Sigmoid(),
            ).to(device)

        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 8, kernel_size=3, stride=2), #255x255 -> 127x127
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2),  #129x129 -> 63x63
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),  #63x63 -> 31x31
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), #31x31 - > 14x14
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2), #14x14 -> 6x6
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2), #6x6 -> 2x2, needs to be output of 2x2 for zdim
                nn.ReLU(),
                Flatten()
            ).to(device)

            self.decoder = nn.Sequential(
                UnFlatten(),
                nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2), #1x1 -> 5x5
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), #5x5 -> 13x13
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), #13x13 -> 29x29
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16,  kernel_size=5, stride=2), #29x29 -> 61x61
                nn.ReLU(),
                nn.ConvTranspose2d(16, 8,  kernel_size=6, stride=2), #61x61 -> 126x126
                nn.ReLU(),
                nn.ConvTranspose2d(8, image_channels,  kernel_size=5, stride=2), #126x126  -> 255x255
                nn.Sigmoid(),
            ).to(device)

        self.fc1 = nn.Linear(h_dim, z_dim) # maps the hidden representation h to the mean of the latent Gaussian distribution
                                           # takes the encoded features as input and produces the mean (mu) of the latent variable z
        self.fc2 = nn.Linear(h_dim, z_dim) # maps the hidden representatiom h to the log-variance of the lantent Gaussian distribution
                                           # produces the log-variance (logvar) of the latent variable z
        self.fc3 = nn.Linear(z_dim, h_dim) # maps the latent sample z to the hidden representation h
                                           # responsible for decoding the sampled latent variable z and generating the reconstructed features 

        self.device = device # device to run the model on

    def reparameterize(self, mu, logvar): # reparameterization trick, facilitates the sampling process in the latent space
        std = logvar.mul(0.5).exp_() # standard deviation of the latent variable z, postive values
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device) # random noise, introduces the element of randomness during sampling
        z = mu + std * esp # reparameterization trick,  allows the model to be trained using backpropagation by making the sampling process differentiable
        return z # returns the latent variable z

    def bottleneck(self, h): # bottleneck layer, combines the mean and the variance of the latent Gaussian distribution
        mu, logvar = self.fc1(h), self.fc2(h) # maps the hidden representation h to the mean and the log-variance of the latent Gaussian distribution
        z = self.reparameterize(mu, logvar) # reparameterization trick, facilitates the sampling process in the latent space
        return z, mu, logvar  
    
    def encode(self, x): # encodes the input features x into the latent space
        h = self.encoder(x) # encodes the input features x into the hidden representation h
        z, mu, logvar = self.bottleneck(h) # combines the mean and the variance of the latent Gaussian distribution
        return z, mu, logvar, h 

    def decode(self, z): # decodes the latent variable z into the output features
        z = self.fc3(z) # maps the latent sample z to the hidden representation h
        z = self.decoder(z) # decodes the hidden representation h into the reconstructed features
        return z # returns the reconstructed features

    def forward(self, x): # forward pass, maps the input features x to the reconstructed features
        h = self.encoder(x) # encodes the input features x into the hidden representation h
        z, mu, logvar = self.bottleneck(h) # combines the mean and the variance of the latent Gaussian distribution
        # z, mu, logvar, _ = self.encode(x)
        recon_x = self.decode(z) # decodes the latent variable z into the reconstructed features
        return recon_x, z, mu, logvar, h

