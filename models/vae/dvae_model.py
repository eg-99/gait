"""
Denoising Variational Autoencoder (DVAE) for GEI Embedding Extraction

Architecture:
- Encoder: Noisy GEI (128×64) → Latent distribution (μ, log_var) in 128-dim space
- Decoder: Sample from latent (128-dim) → Reconstructed clean GEI (128×64)

The model learns to reconstruct clean images from corrupted inputs, making it
more robust to noise, segmentation artifacts, and variations in gait data.

Key differences from standard VAE:
- Inputs are corrupted with noise during training
- Model learns to denoise and reconstruct clean outputs
- More robust embeddings for downstream tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import GEI_VAE, vae_loss


class GEI_DVAE(GEI_VAE):
    """
    Denoising Variational Autoencoder for GEI images.
    
    Inherits from GEI_VAE but adds noise corruption during training.
    Learns robust representations by reconstructing clean images from noisy inputs.
    """
    
    def __init__(self, latent_dim=128, noise_type='gaussian', noise_std=0.1, noise_prob=0.3):
        """
        Initialize Denoising VAE.
        
        Args:
            latent_dim: Dimension of the latent space (default: 128)
            noise_type: Type of noise to apply ('gaussian', 'masking', 'salt_pepper', 'mixed')
            noise_std: Standard deviation for Gaussian noise (default: 0.1)
            noise_prob: Probability for masking/salt-pepper noise (default: 0.3)
        """
        super(GEI_DVAE, self).__init__(latent_dim=latent_dim)
        
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_prob = noise_prob
    
    def add_gaussian_noise(self, x, std=None):
        """
        Add Gaussian noise to input.
        
        Args:
            x: Input tensor (batch, 1, 128, 64)
            std: Standard deviation (uses self.noise_std if None)
        
        Returns:
            Corrupted tensor with same shape
        """
        if std is None:
            std = self.noise_std
        
        noise = torch.randn_like(x) * std
        corrupted = x + noise
        
        # Clamp to [0, 1] range
        corrupted = torch.clamp(corrupted, 0.0, 1.0)
        
        return corrupted
    
    def add_masking_noise(self, x, prob=None):
        """
        Randomly mask out pixels (set to 0).
        
        Args:
            x: Input tensor (batch, 1, 128, 64)
            prob: Probability of masking each pixel (uses self.noise_prob if None)
        
        Returns:
            Corrupted tensor with same shape
        """
        if prob is None:
            prob = self.noise_prob
        
        mask = torch.rand_like(x) > prob
        corrupted = x * mask.float()
        
        return corrupted
    
    def add_salt_pepper_noise(self, x, prob=None):
        """
        Add salt and pepper noise (random pixels set to 0 or 1).
        
        Args:
            x: Input tensor (batch, 1, 128, 64)
            prob: Probability of corrupting each pixel (uses self.noise_prob if None)
        
        Returns:
            Corrupted tensor with same shape
        """
        if prob is None:
            prob = self.noise_prob
        
        # Random mask for corruption
        corruption_mask = torch.rand_like(x) < prob
        
        # Randomly assign salt (1) or pepper (0)
        salt_pepper = torch.rand_like(x) > 0.5
        
        corrupted = x.clone()
        corrupted[corruption_mask] = salt_pepper[corruption_mask].float()
        
        return corrupted
    
    def add_mixed_noise(self, x):
        """
        Apply a random combination of noise types.
        
        Args:
            x: Input tensor (batch, 1, 128, 64)
        
        Returns:
            Corrupted tensor with same shape
        """
        # Randomly choose noise type
        noise_choice = torch.rand(1).item()
        
        if noise_choice < 0.4:
            # Gaussian noise
            return self.add_gaussian_noise(x)
        elif noise_choice < 0.7:
            # Masking noise
            return self.add_masking_noise(x)
        else:
            # Salt and pepper
            return self.add_salt_pepper_noise(x)
    
    def corrupt_input(self, x, training=True):
        """
        Corrupt input based on noise type.
        
        Args:
            x: Input tensor (batch, 1, 128, 64)
            training: Whether in training mode (only corrupt during training)
        
        Returns:
            Corrupted tensor (or original if not training)
        """
        if not training:
            # During inference, optionally return clean or corrupted
            # For now, return clean during inference
            return x
        
        if self.noise_type == 'gaussian':
            return self.add_gaussian_noise(x)
        elif self.noise_type == 'masking':
            return self.add_masking_noise(x)
        elif self.noise_type == 'salt_pepper':
            return self.add_salt_pepper_noise(x)
        elif self.noise_type == 'mixed':
            return self.add_mixed_noise(x)
        else:
            # Unknown noise type, return original
            return x
    
    def forward(self, x, corrupt=True):
        """
        Full DVAE forward pass.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64) - clean images
            corrupt: Whether to corrupt input (default: True for training)
        
        Returns:
            reconstruction: Reconstructed clean GEI (batch, 1, 128, 64)
            mu: Latent mean (batch, latent_dim)
            log_var: Latent log variance (batch, latent_dim)
            corrupted: Corrupted input (batch, 1, 128, 64) - for visualization
        """
        # Corrupt input during training
        corrupted = self.corrupt_input(x, training=corrupt)
        
        # Encode corrupted input
        mu, log_var = self.encode(corrupted)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode to reconstruct clean image
        reconstruction = self.decode(z)
        
        return reconstruction, mu, log_var, corrupted
    
    def forward_clean(self, x):
        """
        Forward pass without corruption (for inference/embedding extraction).
        
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


def dvae_loss(reconstruction, target, mu, log_var, corrupted=None, beta=1.0, denoising_weight=1.0):
    """
    Compute DVAE loss = Reconstruction Loss + β * KL Divergence.
    
    The reconstruction loss compares the output to the CLEAN target,
    encouraging the model to denoise corrupted inputs.
    
    Args:
        reconstruction: Reconstructed images (batch, 1, 128, 64)
        target: Original CLEAN images (batch, 1, 128, 64)
        mu: Latent mean (batch, latent_dim)
        log_var: Latent log variance (batch, latent_dim)
        corrupted: Corrupted input images (optional, for visualization)
        beta: Weight for KL divergence term (default: 1.0)
        denoising_weight: Weight for denoising loss (default: 1.0, same as standard VAE)
    
    Returns:
        total_loss: Total DVAE loss
        recon_loss: Reconstruction loss component (denoising loss)
        kl_loss: KL divergence component
    """
    # Reconstruction loss (MSE) - compare to CLEAN target
    recon_loss = F.mse_loss(reconstruction, target, reduction='sum')
    recon_loss = recon_loss / target.size(0)  # Average over batch
    recon_loss = recon_loss * denoising_weight
    
    # KL divergence: KL(N(μ, σ²) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss / target.size(0)  # Average over batch
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def create_dvae(latent_dim=128, noise_type='gaussian', noise_std=0.1, noise_prob=0.3):
    """
    Factory function to create Denoising VAE with weight initialization.
    
    Args:
        latent_dim: Dimension of latent space
        noise_type: Type of noise ('gaussian', 'masking', 'salt_pepper', 'mixed')
        noise_std: Standard deviation for Gaussian noise
        noise_prob: Probability for masking/salt-pepper noise
    
    Returns:
        Initialized DVAE model
    """
    model = GEI_DVAE(
        latent_dim=latent_dim,
        noise_type=noise_type,
        noise_std=noise_std,
        noise_prob=noise_prob
    )
    
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
    print("Testing Denoising VAE...")
    
    # Test with different noise types
    noise_types = ['gaussian', 'masking', 'salt_pepper', 'mixed']
    
    for noise_type in noise_types:
        print(f"\n{'='*60}")
        print(f"Testing with noise type: {noise_type}")
        print(f"{'='*60}")
        
        model = create_dvae(
            latent_dim=128,
            noise_type=noise_type,
            noise_std=0.1,
            noise_prob=0.3
        )
        
        # Test with dummy input
        batch_size = 4
        dummy_input = torch.rand(batch_size, 1, 128, 64)  # [0, 1] range
        
        # Forward pass with corruption
        reconstruction, mu, log_var, corrupted = model(dummy_input, corrupt=True)
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Corrupted shape: {corrupted.shape}")
        print(f"Mu shape: {mu.shape}")
        print(f"Log_var shape: {log_var.shape}")
        print(f"Reconstruction shape: {reconstruction.shape}")
        
        # Test loss
        total_loss, recon_loss, kl_loss = dvae_loss(
            reconstruction, dummy_input, mu, log_var, corrupted
        )
        print(f"\nLoss components:")
        print(f"  Reconstruction (denoising) loss: {recon_loss.item():.6f}")
        print(f"  KL divergence: {kl_loss.item():.6f}")
        print(f"  Total loss: {total_loss.item():.6f}")
        
        # Test forward_clean (no corruption)
        recon_clean, mu_clean, log_var_clean = model.forward_clean(dummy_input)
        print(f"\nForward clean - Reconstruction shape: {recon_clean.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test sampling
        samples = model.sample(num_samples=8, device='cpu')
        print(f"Generated samples shape: {samples.shape}")
    
    print(f"\n{'='*60}")
    print("All tests completed!")

