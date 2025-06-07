import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

# Create the vector of filter multipliers based on the current resolution, these multipliers are applied to 256 filters (default number of filters)
nf = [1, 1, 1, 1, 1/2, 1/4]

# Linear layer with weight scaling whose output is (batch size, 256, 4, 4)
class LinearInicial(nn.Module):
    def __init__(self, inputs, outputs, gain=2):
        super(LinearInicial, self).__init__()
        # Initialize the linear layer
        self.linear = nn.Linear(inputs, outputs)
        # Set how the weights will be rescaled
        self.equalized = gain / inputs ** 0.5
        self.in_features = inputs
        # Initialize the weights
        nn.init.normal_(self.linear.weight)

    def forward(self, x):
        # Transform the input vector to be suitable for the layer
        x = x.view(x.size(0), -1)
        # Return the result of applying the linear layer to the input vector
        return self.linear(x * self.equalized).view(-1, self.in_features, 4, 4)

# Linear layer with weight scaling whose output is two-dimensional
class Linear(nn.Module):
    def __init__(self, inputs, outputs, gain=2):
        super(Linear, self).__init__()
        # Initialize the linear layer
        self.linear = nn.Linear(inputs, outputs)
        # Set how the weights will be rescaled
        self.equalized = gain / inputs ** 0.5
        # Initialize the weights and biases
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Transform the input vector to be suitable for the layer
        x = x.view(x.size(0), -1)
        # Return the result of applying the linear layer to the input vector
        return self.linear(x * self.equalized)

# Convolutional layer with weight scaling
class Conv2d(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2
    ):
        super(Conv2d, self).__init__()
        # Initialize the convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # Set how the weights will be rescaled
        self.equalized = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        # Set the convolutional biases to not be updated (only weights are updated)
        self.bias = self.conv.bias
        self.conv.bias = None
        # Initialize the weights and copy of the biases
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Return the result of applying the convolutional layer to the input vector plus the copy of the biases
        return self.conv(x * self.equalized) + self.bias.view(1, self.bias.shape[0], 1, 1)

# Create the Pixelwise Normalization layer
class PN(nn.Module):
    def __init__(self):
        super(PN, self).__init__()
        # Initialize epsilon to 10^-8
        self.epsilon = 1e-8
    def forward(self, x):
        # Return the result of applying Pixelwise Normalization to the input vector
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

# Create a convolutional block to "package" the concatenation of two convolutional blocks with Leaky ReLU and Pixelwise Normalization
class BloqueConvolucional(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(BloqueConvolucional, self).__init__()
        # Initialize the variable to check if Pixelwise Normalization should be used
        self.use_pn = use_pixelnorm
        # Initialize the first convolutional layer
        self.conv1 = Conv2d(in_channels, out_channels)
        # Initialize the second convolutional layer
        self.conv2 = Conv2d(out_channels, out_channels)
        # Set the activation function to LeakyReLU
        self.leaky = nn.LeakyReLU(0.2)
        # Initialize the Pixelwise Normalization layer
        self.pn = PN()

    def forward(self, x):
        # Apply the Leaky ReLU function to the output of the first convolutional layer
        x = self.leaky(self.conv1(x))
        # Apply Pixelwise Normalization in the case of generator convolutions to conv1
        x = self.pn(x) if self.use_pn else x
        # Apply the Leaky ReLU function to the output of the second convolutional layer
        x = self.leaky(self.conv2(x))
        # Apply Pixelwise Normalization in the case of generator convolutions to conv2
        x = self.pn(x) if self.use_pn else x
        return x

# Build the generator
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels=3):
        super(Generator, self).__init__()
        # Initialize the first block (4x4 resolution) of the proGAN generator
        self.bloqueInicial = nn.Sequential(
            # Apply a Pixelwise Normalization layer to the generator input
            PN(),
            # Then apply a linear layer with Leaky ReLU activation function
            LinearInicial(latent_dim, latent_dim * 16),
            nn.LeakyReLU(0.2),
            # Apply the first convolutional layer with Leaky ReLU activation function
            Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # Apply Pixelwise Normalization to the output of the convolutional layer
            PN(),
        )

        # Initialize the generator layer that transforms the generator output to RGB (CN RGB)
        self.cn_RGB = Conv2d(latent_dim, img_channels, kernel_size=1, stride=1, padding=0)

        # Initialize the modules that will contain all the remaining generator blocks (resBloques) and their respective blocks that transform their outputs to RGB (cn_RGB_layers)
        self.resBloques, self.cn_RGB_layers = (nn.ModuleList([]), nn.ModuleList([self.cn_RGB]))

        # Initialize the rest of the generator blocks (8x8, 16x16, ..., 128x128)
        for i in range(len(nf) - 1):
            # Specify the number of input filters expected by the different blocks of the different resolutions
            filtrosEntrada = int(latent_dim * nf[i])
            # Specify the number of output filters that the different blocks of the different resolutions should return
            filtrosSalida = int(latent_dim * nf[i + 1])
            # Add the new block to the generator blocks module
            self.resBloques.append(BloqueConvolucional(filtrosEntrada, filtrosSalida))
            # Add the new CN RGB layer to the CN RGB layers module
            self.cn_RGB_layers.append(Conv2d(filtrosSalida, img_channels, kernel_size=1, stride=1, padding=0))

    # Define the function that fades the generator output
    def fade_in(self, alpha, upsampled, lastBlock):
        return torch.tanh(alpha * lastBlock + (1 - alpha) * upsampled)

    def forward(self, x, alpha, steps):
        # Apply the first 4x4 block to the generator input
        output = self.bloqueInicial(x)

        # If the current resolution is 4x4, apply bloqueInicial to the input and transform its output to RGB without fading
        if steps == 0:
            return self.cn_RGB(output)

        # Check the current resolution and apply the rest of the blocks after the output of bloqueInicial until reaching the current resolution
        for step in range(steps):
            # At the beginning of each block of the proGAN generator, apply an upsampling layer using Nearest Neighbors with a scale factor of 2x
            upsampled = F.interpolate(output, scale_factor=2, mode="nearest")
            # Apply the convolutional block (two convolutions with Leaky ReLU activation and Pixelwise Normalization) to the output of the upsampling layer
            output = self.resBloques[step](upsampled)

        # After applying all blocks to the generator input, two branches are produced: one where the output of the last upsampling layer is transformed to RGB
        lastUpsampled = self.cn_RGB_layers[steps - 1](upsampled)
        # Another where the output of the last convolutional block is transformed to RGB
        lastBlock = self.cn_RGB_layers[steps](output)
        # Return the two branches faded
        return self.fade_in(alpha, lastUpsampled, lastBlock)

# Build the discriminator
class Discriminator(nn.Module):
    def __init__(self, latent_dim, img_channels=3):
        super(Discriminator, self).__init__()
        # Initialize the modules that will contain all the discriminator blocks (resBloques), except for the 4x4 one, and their respective previous blocks that transform the discriminator input to NOT RGB (cn_NOT_RGB_layers)
        self.resBloques, self.cn_NOT_RGB_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # Initialize all discriminator blocks (except the 4x4 one)
        for i in range(len(nf) - 1, 0, -1):
            # Specify the number of input filters expected by the different blocks of the different resolutions
            filtrosEntrada = int(latent_dim * nf[i])
            # Specify the number of output filters that the different blocks of the different resolutions should return
            filtrosSalida = int(latent_dim * nf[i - 1])
            # Add the new block to the discriminator blocks module, in this case Pixelwise Normalization is not used
            self.resBloques.append(BloqueConvolucional(filtrosEntrada, filtrosSalida, use_pixelnorm=False))
            # Add the new CN NOT RGB layer to the CN NOT RGB layers module
            self.cn_NOT_RGB_layers.append(Conv2d(img_channels, filtrosEntrada, kernel_size=1, stride=1, padding=0))
        # Add the last CN NOT RGB layer to the CN NOT RGB layers module
        self.cn_NOT_RGB_layers.append(Conv2d(img_channels, latent_dim, kernel_size=1, stride=1, padding=0))

        # Initialize the Average Pooling layer that decreases the resolution of the discriminator input as it advances through the network
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # down sampling using avg pool

        # Initialize the last block (4x4 resolution) of the proGAN discriminator
        self.bloqueFinal = nn.Sequential(
            # Apply a convolutional layer with Leaky ReLU activation that transforms its input from (257x4x4) to (256x4x4)
            Conv2d(latent_dim + 1, latent_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            # Then apply a linear layer with Leaky ReLU activation that "flattens" its input
            Linear(latent_dim*16, latent_dim),
            nn.LeakyReLU(0.2),
            # Finally, apply a linear layer with Leaky ReLU activation that produces an output (the discriminator output) of dimensions (batch size, 1)
            Linear(latent_dim, 1),
        )

    # Define the function that fades the discriminator input
    def fade_in(self, alpha, downsampled, output):
        return alpha * output + (1 - alpha) * downsampled

    # Define the Mini-batch layer
    def minibatch_std(self, x):
        batchStats = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batchStats], dim=1)

    def forward(self, x, alpha, steps):
        # Create a variable that contains the current resolution being worked with
        currentRes = len(self.resBloques) - steps
        # Apply the first CN NOT RGB layer to the discriminator input
        output = self.leaky(self.cn_NOT_RGB_layers[currentRes](x))

        # Check if the resolution being worked with is 4x4 and if so, apply the Mini-batch layer and bloqueFinal to "output"
        if steps == 0:  # i.e, image is 4x4
            output = self.minibatch_std(output)
            return self.bloqueFinal(output).view(output.shape[0], -1)

        # Apply fading to the two branches of the discriminator
        downsampled = self.leaky(self.cn_NOT_RGB_layers[currentRes + 1](self.avg_pool(x)))
        output = self.avg_pool(self.resBloques[currentRes](output))
        output = self.fade_in(alpha, downsampled, output)

        # Check the current resolution and apply the rest of the blocks after fading the discriminator input until reaching the 4x4 resolution
        for step in range(currentRes + 1, len(self.resBloques)):
            output = self.avg_pool(self.resBloques[step](output))

        # Once the 4x4 resolution is reached, apply the Mini-batch layer
        output = self.minibatch_std(output)
        # Apply bloqueFinal to the output of the Mini-batch layer
        return self.bloqueFinal(output).view(output.shape[0], -1)
