# Generative AI

A collection of Jupyter notebooks and scripts implementing different generative models in TensorFlow/Keras and PyTorch, including Variational Autoencoders (VAE), Generative Adversarial Networks (GANs) and Denoising Diffusion Probabilistic Models (DDPM).

## Repository Structure
```
Generative AI/
├── VAE/
│   └── VAE.ipynb                  # Variational Autoencoder notebook (TensorFlow/Keras)
├── GANS/
│   ├── DCGAN/
│   │   └── DCGAN.ipynb            # Deep Convolutional GAN notebook (TensorFlow/Keras)
│   ├── GAN/
│   │   └── GAN.ipynb              # Vanilla GAN notebook (TensorFlow/Keras)
│   └── proGAN/
│       ├── model.py               # Progressive GAN model definition (PyTorch)
│       └── train.py               # proGAN training script (PyTorch)
└── DDPM/
    ├── DDPM linear.ipynb          # Linear noise schedule diffusion notebook (TensorFlow/Keras)
    └── DDPM sinusoidal.ipynb      # Sinusoidal noise schedule diffusion notebook (TensorFlow/Keras)
```

## Models
### 1. Variational Autoencoder (VAE)
- **Variational Autoencoder** (`VAE/VAE.ipynb`): A generative model that learns a probabilistic latent space using the reparameterization trick and optimizes a KL divergence + MSE loss.

### 2. Generative Adversarial Networks (GANs)
- **Deep Convolutional Generative Adversarial Network (DCGAN)** (`GANS/DCGAN/DCGAN.ipynb`): A GAN that uses convolutional layers.  
- **Vanilla GAN (GAN)** (`GANS/GAN/GAN.ipynb`): The standard GAN (TensorFlow/Keras).  
- **Progressive Growing of GAN (proGAN)** (`GANS/proGAN/`): A GAN that "grows" during training (more layers are involved as training progresses).  

### 3. Denoising Diffusion Probabilistic Models (DDPM)
- **Linear schedule** (`DDPM/DDPM linear.ipynb`): A generative model that learns to reverse a noising process using a U-Net denoiser with a linear variance schedule.  
- **Sinusoidal schedule** (`DDPM/DDPM sinusoidal.ipynb`): Similar to the linear version but uses a sinusoidal noise schedule for improved performance in some cases.  


## Tested With
- TensorFlow: 2.12.0  
- PyTorch: 2.0.1  

## Requirements
```bash
pip install tensorflow==2.12.0 torch==2.0.1 torchvision==0.15.2 matplotlib numpy tqdm tensorflow-addons
```

## Usage
1. Clone this repo.  
2. Open the code in the preferred IDE.  
3. In the first cell or at the top of each script, set:  
   - `directory` – path to your image dataset  
   - `trained_models_folder` – where to save model weights  
   - `generated_images_folder` – where to save output images  
4. Run all cells (or `python GANS/proGAN/train.py`) to start training and logging to TensorBoard:
   ```bash
   tensorboard --logdir logs/
   ```
