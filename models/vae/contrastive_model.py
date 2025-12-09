"""
Contrastive Learning Model for GEI Embeddings

Uses VAE encoder as backbone and adds projection head for contrastive learning.
Combines VAE reconstruction with contrastive learning for robust embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GEI_VAE


class ContrastiveGEIEncoder(nn.Module):
    """
    Contrastive learning encoder using VAE encoder as backbone.
    
    Extracts embeddings from GEI images and projects them to a space
    suitable for contrastive learning.
    """
    
    def __init__(self, vae_model, embedding_dim=128, projection_dim=128):
        """
        Initialize contrastive encoder.
        
        Args:
            vae_model: Pre-trained or untrained VAE model (we use its encoder)
            embedding_dim: Dimension of VAE latent space (default: 128)
            projection_dim: Dimension of projection head output (default: 128)
        """
        super(ContrastiveGEIEncoder, self).__init__()
        
        # Use VAE encoder as backbone
        self.encoder = vae_model
        
        # Projection head for contrastive learning
        # Maps from embedding_dim to projection_dim
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )
        
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
    
    def encode(self, x):
        """
        Encode GEI image to latent distribution.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64)
        
        Returns:
            mu: Latent mean (batch, embedding_dim)
            log_var: Latent log variance (batch, embedding_dim)
        """
        return self.encoder.encode(x)
    
    def get_embedding(self, x):
        """
        Get embedding (mean) from GEI image.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64)
        
        Returns:
            embedding: Latent mean (batch, embedding_dim)
        """
        mu, _ = self.encode(x)
        return mu
    
    def forward(self, x, return_projection=True):
        """
        Forward pass through encoder and projection head.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64)
            return_projection: If True, return projected embedding; else return raw embedding
        
        Returns:
            If return_projection:
                projection: Projected embedding (batch, projection_dim)
            Else:
                embedding: Raw embedding (batch, embedding_dim)
        """
        mu, log_var = self.encode(x)
        
        if return_projection:
            projection = self.projection_head(mu)
            return projection, mu, log_var
        else:
            return mu, log_var


class ContrastiveVAE(nn.Module):
    """
    Combined VAE + Contrastive Learning model.
    
    Trains VAE with both reconstruction loss and contrastive loss
    to learn robust embeddings.
    """
    
    def __init__(self, latent_dim=128, projection_dim=128):
        """
        Initialize Contrastive VAE.
        
        Args:
            latent_dim: Dimension of VAE latent space (default: 128)
            projection_dim: Dimension of projection head output (default: 128)
        """
        super(ContrastiveVAE, self).__init__()
        
        # VAE backbone
        self.vae = GEI_VAE(latent_dim=latent_dim)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, projection_dim)
        )
        
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim
    
    def encode(self, x):
        """Encode GEI to latent distribution."""
        return self.vae.encode(x)
    
    def decode(self, z):
        """Decode latent to GEI."""
        return self.vae.decode(z)
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        return self.vae.reparameterize(mu, log_var)
    
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
    
    def forward(self, x, return_projection=False):
        """
        Full forward pass.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64)
            return_projection: If True, also return projection for contrastive loss
        
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
        return self.vae.sample(num_samples, device)


def create_contrastive_vae(latent_dim=128, projection_dim=128):
    """
    Factory function to create Contrastive VAE.
    
    Args:
        latent_dim: Dimension of VAE latent space
        projection_dim: Dimension of projection head output
    
    Returns:
        Initialized Contrastive VAE model
    """
    model = ContrastiveVAE(latent_dim=latent_dim, projection_dim=projection_dim)
    
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
    model = create_contrastive_vae(latent_dim=128, projection_dim=128)
    print(model)
    
    # Test with dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 128, 64)
    
    # Test forward pass
    reconstruction, mu, log_var, projection = model(dummy_input, return_projection=True)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log_var shape: {log_var.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Projection shape: {projection.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")



