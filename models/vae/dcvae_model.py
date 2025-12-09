"""
Denoising Contrastive Variational Autoencoder (D-CVAE) for GEI Embedding Extraction

Combines Denoising VAE with Contrastive Learning:
- Denoising: Learns to reconstruct clean images from corrupted inputs (robustness)
- Contrastive: Learns discriminative embeddings by pulling similar samples together (discriminative power)

Architecture:
- Encoder: Noisy GEI (128×64) → Latent distribution (μ, log_var) in 128-dim space
- Decoder: Sample from latent (128-dim) → Reconstructed clean GEI (128×64)
- Projection Head: Latent mean (128-dim) → Projected embedding (128-dim) for contrastive learning

The model learns both robust (denoising) and discriminative (contrastive) representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dvae_model import GEI_DVAE, dvae_loss
from .contrastive_loss import info_nce_loss


class DenoisingContrastiveVAE(nn.Module):
    """
    Denoising Contrastive Variational Autoencoder for GEI images.
    
    Combines denoising (robustness) with contrastive learning (discriminative power).
    """
    
    def __init__(self, latent_dim=128, projection_dim=128, 
                 noise_type='gaussian', noise_std=0.1, noise_prob=0.3):
        """
        Initialize Denoising Contrastive VAE.
        
        Args:
            latent_dim: Dimension of VAE latent space (default: 128)
            projection_dim: Dimension of projection head output (default: 128)
            noise_type: Type of noise to apply ('gaussian', 'masking', 'salt_pepper', 'mixed')
            noise_std: Standard deviation for Gaussian noise (default: 0.1)
            noise_prob: Probability for masking/salt-pepper noise (default: 0.3)
        """
        super(DenoisingContrastiveVAE, self).__init__()
        
        # DVAE backbone (provides denoising capability)
        self.dvae = GEI_DVAE(
            latent_dim=latent_dim,
            noise_type=noise_type,
            noise_std=noise_std,
            noise_prob=noise_prob
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, projection_dim)
        )
        
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_prob = noise_prob
    
    def encode(self, x):
        """Encode GEI to latent distribution."""
        return self.dvae.encode(x)
    
    def decode(self, z):
        """Decode latent to GEI."""
        return self.dvae.decode(z)
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        return self.dvae.reparameterize(mu, log_var)
    
    def get_embedding(self, x):
        """Get embedding (mean) from GEI."""
        mu, _ = self.encode(x)
        return mu
    
    def get_projection(self, x):
        """
        Get projected embedding for contrastive learning.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64)
        
        Returns:
            projection: Projected embedding (batch, projection_dim)
            mu: Latent mean (batch, latent_dim)
            log_var: Latent log variance (batch, latent_dim)
        """
        mu, log_var = self.encode(x)
        projection = self.projection_head(mu)
        return projection, mu, log_var
    
    def forward(self, x, corrupt=True, return_projection=False):
        """
        Full forward pass with denoising and optional contrastive projection.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64) - clean images
            corrupt: Whether to corrupt input (default: True for training)
            return_projection: If True, also return projection for contrastive loss
        
        Returns:
            If return_projection:
                reconstruction, mu, log_var, corrupted, projection
            Else:
                reconstruction, mu, log_var, corrupted
        """
        # Forward through DVAE (with corruption)
        reconstruction, mu, log_var, corrupted = self.dvae(x, corrupt=corrupt)
        
        if return_projection:
            projection = self.projection_head(mu)
            return reconstruction, mu, log_var, corrupted, projection
        else:
            return reconstruction, mu, log_var, corrupted
    
    def forward_clean(self, x, return_projection=False):
        """
        Forward pass without corruption (for inference/embedding extraction).
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64)
            return_projection: If True, also return projection
        
        Returns:
            If return_projection:
                reconstruction, mu, log_var, projection
            Else:
                reconstruction, mu, log_var
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        
        if return_projection:
            projection = self.projection_head(mu)
            return reconstruction, mu, log_var, projection
        else:
            return reconstruction, mu, log_var
    
    def sample(self, num_samples, device):
        """Generate samples from prior."""
        return self.dvae.sample(num_samples, device)


def dcvae_loss(reconstruction, target, mu, log_var, corrupted=None, 
               projection=None, labels=None, 
               beta=1.0, contrastive_weight=1.0, temperature=0.07):
    """
    Compute D-CVAE loss = Denoising Reconstruction Loss + β * KL Divergence + λ * Contrastive Loss.
    
    Args:
        reconstruction: Reconstructed images (batch, 1, 128, 64)
        target: Original CLEAN images (batch, 1, 128, 64)
        mu: Latent mean (batch, latent_dim)
        log_var: Latent log variance (batch, latent_dim)
        corrupted: Corrupted input images (optional, for visualization)
        projection: Projected embeddings for contrastive learning (batch, projection_dim)
        labels: Subject labels for contrastive loss (batch,)
        beta: Weight for KL divergence term (default: 1.0)
        contrastive_weight: Weight for contrastive loss term (default: 1.0)
        temperature: Temperature for contrastive loss (default: 0.07)
    
    Returns:
        total_loss: Total D-CVAE loss
        recon_loss: Denoising reconstruction loss component
        kl_loss: KL divergence component
        contrastive_loss: Contrastive loss component (0 if projection/labels not provided)
    """
    # Denoising reconstruction loss (compare to CLEAN target)
    recon_loss = F.mse_loss(reconstruction, target, reduction='sum')
    recon_loss = recon_loss / target.size(0)  # Average over batch
    
    # KL divergence: KL(N(μ, σ²) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss / target.size(0)  # Average over batch
    
    # Contrastive loss (if projection and labels provided)
    contrastive_loss = torch.tensor(0.0, device=reconstruction.device)
    if projection is not None and labels is not None:
        contrastive_loss = info_nce_loss(projection, labels, temperature=temperature)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss + contrastive_weight * contrastive_loss
    
    return total_loss, recon_loss, kl_loss, contrastive_loss


def create_dcvae(latent_dim=128, projection_dim=128,
                 noise_type='gaussian', noise_std=0.1, noise_prob=0.3):
    """
    Factory function to create Denoising Contrastive VAE with weight initialization.
    
    Args:
        latent_dim: Dimension of latent space
        projection_dim: Dimension of projection head output
        noise_type: Type of noise ('gaussian', 'masking', 'salt_pepper', 'mixed')
        noise_std: Standard deviation for Gaussian noise
        noise_prob: Probability for masking/salt-pepper noise
    
    Returns:
        Initialized D-CVAE model
    """
    model = DenoisingContrastiveVAE(
        latent_dim=latent_dim,
        projection_dim=projection_dim,
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
    print("Testing Denoising Contrastive VAE...")
    
    model = create_dcvae(
        latent_dim=128,
        projection_dim=128,
        noise_type='gaussian',
        noise_std=0.1,
        noise_prob=0.3
    )
    
    # Test with dummy input
    batch_size = 4
    dummy_input = torch.rand(batch_size, 1, 128, 64)  # [0, 1] range
    dummy_labels = torch.randint(0, 2, (batch_size,))  # Subject labels
    
    # Forward pass with corruption and projection
    reconstruction, mu, log_var, corrupted, projection = model(
        dummy_input, corrupt=True, return_projection=True
    )
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Corrupted shape: {corrupted.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log_var shape: {log_var.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Projection shape: {projection.shape}")
    
    # Test loss
    total_loss, recon_loss, kl_loss, contrastive_loss = dcvae_loss(
        reconstruction, dummy_input, mu, log_var, 
        corrupted=corrupted, projection=projection, labels=dummy_labels
    )
    print(f"\nLoss components:")
    print(f"  Denoising reconstruction loss: {recon_loss.item():.6f}")
    print(f"  KL divergence: {kl_loss.item():.6f}")
    print(f"  Contrastive loss: {contrastive_loss.item():.6f}")
    print(f"  Total loss: {total_loss.item():.6f}")
    
    # Test forward_clean
    recon_clean, mu_clean, log_var_clean, proj_clean = model.forward_clean(
        dummy_input, return_projection=True
    )
    print(f"\nForward clean - Reconstruction shape: {recon_clean.shape}")
    print(f"Forward clean - Projection shape: {proj_clean.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test sampling
    samples = model.sample(num_samples=8, device='cpu')
    print(f"Generated samples shape: {samples.shape}")

