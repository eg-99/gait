"""
Variational Autoencoder (VAE) for GEI Embedding Extraction

Architecture:
- Encoder: GEI (128×64) → Latent distribution (μ, log_var) in 128-dim space
- Decoder: Sample from latent (128-dim) → Reconstructed GEI (128×64)

The 128-dim latent mean (μ) serves as the embedding space.
Uses reparameterization trick for backpropagation through sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GEI_VAE(nn.Module):
    """
    Variational Autoencoder for GEI images.
    
    Learns a probabilistic latent space with mean (μ) and variance (σ²).
    The mean vector serves as the embedding for each GEI.
    """
    
    def __init__(self, latent_dim=128):
        """
        Initialize VAE.
        
        Args:
            latent_dim: Dimension of the latent space (default: 128)
        """
        super(GEI_VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # ====== ENCODER ======
        # Input: (batch, 1, 128, 64)
        
        # Conv Block 1: 1 → 32 channels
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # → (32, 64, 32)
        self.enc_bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2: 32 → 64 channels
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # → (64, 32, 16)
        self.enc_bn2 = nn.BatchNorm2d(64)
        
        # Conv Block 3: 64 → 128 channels
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # → (128, 16, 8)
        self.enc_bn3 = nn.BatchNorm2d(128)
        
        # Conv Block 4: 128 → 256 channels
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # → (256, 8, 4)
        self.enc_bn4 = nn.BatchNorm2d(256)
        
        # Flatten: 256 * 8 * 4 = 8192
        self.flatten_size = 256 * 8 * 4
        
        # Fully connected to latent parameters
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)  # Mean
        self.fc_log_var = nn.Linear(self.flatten_size, latent_dim)  # Log variance
        
        # ====== DECODER ======
        # Latent → Reconstruct
        
        self.dec_fc = nn.Linear(latent_dim, self.flatten_size)
        
        # ConvTranspose Block 1: 256 → 128 channels
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # → (128, 16, 8)
        self.dec_bn1 = nn.BatchNorm2d(128)
        
        # ConvTranspose Block 2: 128 → 64 channels
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # → (64, 32, 16)
        self.dec_bn2 = nn.BatchNorm2d(64)
        
        # ConvTranspose Block 3: 64 → 32 channels
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # → (32, 64, 32)
        self.dec_bn3 = nn.BatchNorm2d(32)
        
        # ConvTranspose Block 4: 32 → 1 channel (final reconstruction)
        self.dec_conv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)  # → (1, 128, 64)
    
    def encode(self, x):
        """
        Encode GEI image to latent distribution parameters.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64)
        
        Returns:
            mu: Mean of latent distribution (batch, latent_dim)
            log_var: Log variance of latent distribution (batch, latent_dim)
        """
        # Encoder forward pass
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = F.relu(self.enc_bn3(self.enc_conv3(x)))
        x = F.relu(self.enc_bn4(self.enc_conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Latent parameters
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: sample z from N(μ, σ²).
        
        z = μ + σ * ε, where ε ~ N(0, 1)
        
        Args:
            mu: Mean (batch, latent_dim)
            log_var: Log variance (batch, latent_dim)
        
        Returns:
            z: Sampled latent vector (batch, latent_dim)
        """
        std = torch.exp(0.5 * log_var)  # σ = exp(0.5 * log_var)
        eps = torch.randn_like(std)  # ε ~ N(0, 1)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector back to GEI image.
        
        Args:
            z: Latent vector (batch, latent_dim)
        
        Returns:
            Reconstructed GEI tensor (batch, 1, 128, 64)
        """
        # Expand latent
        x = self.dec_fc(z)
        
        # Reshape to spatial dimensions
        x = x.view(x.size(0), 256, 8, 4)
        
        # Decoder forward pass
        x = F.relu(self.dec_bn1(self.dec_conv1(x)))
        x = F.relu(self.dec_bn2(self.dec_conv2(x)))
        x = F.relu(self.dec_bn3(self.dec_conv3(x)))
        x = torch.sigmoid(self.dec_conv4(x))  # Sigmoid for [0,1] range
        
        return x
    
    def forward(self, x):
        """
        Full VAE forward pass.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64)
        
        Returns:
            reconstruction: Reconstructed GEI (batch, 1, 128, 64)
            mu: Latent mean (batch, latent_dim)
            log_var: Latent log variance (batch, latent_dim)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, log_var
    
    def sample(self, num_samples, device):
        """
        Generate new samples by sampling from standard normal latent space.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            Generated GEI images (num_samples, 1, 128, 64)
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


def vae_loss(reconstruction, target, mu, log_var, beta=1.0):
    """
    Compute VAE loss = Reconstruction Loss + β * KL Divergence.
    
    Args:
        reconstruction: Reconstructed images (batch, 1, 128, 64)
        target: Original images (batch, 1, 128, 64)
        mu: Latent mean (batch, latent_dim)
        log_var: Latent log variance (batch, latent_dim)
        beta: Weight for KL divergence term (default: 1.0)
    
    Returns:
        total_loss: Total VAE loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, target, reduction='sum')
    recon_loss = recon_loss / target.size(0)  # Average over batch
    
    # KL divergence: KL(N(μ, σ²) || N(0, 1))
    # = -0.5 * sum(1 + log_var - mu² - exp(log_var))
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss / target.size(0)  # Average over batch
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def create_vae(latent_dim=128):
    """
    Factory function to create VAE with weight initialization.
    
    Args:
        latent_dim: Dimension of latent space
    
    Returns:
        Initialized VAE model
    """
    model = GEI_VAE(latent_dim=latent_dim)
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_vae(latent_dim=128)
    print(model)
    
    # Test with dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 128, 64)
    
    reconstruction, mu, log_var = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log_var shape: {log_var.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Test loss
    total_loss, recon_loss, kl_loss = vae_loss(reconstruction, dummy_input, mu, log_var)
    print(f"\nLoss components:")
    print(f"  Reconstruction loss: {recon_loss.item():.6f}")
    print(f"  KL divergence: {kl_loss.item():.6f}")
    print(f"  Total loss: {total_loss.item():.6f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test sampling
    samples = model.sample(num_samples=8, device='cpu')
    print(f"\nGenerated samples shape: {samples.shape}")
