"""
Standard Autoencoder for GEI Embedding Extraction

Architecture:
- Encoder: GEI (128×64) → Bottleneck (128-dim embedding)
- Decoder: Embedding (128-dim) → Reconstructed GEI (128×64)

The 128-dim bottleneck serves as the embedding space for gait representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GEI_Autoencoder(nn.Module):
    """
    Standard Autoencoder for GEI images.
    
    Compresses GEI (128×64 = 8192 pixels) into 128-dimensional embedding,
    then reconstructs back to original size.
    """
    
    def __init__(self, embedding_dim=128):
        """
        Initialize autoencoder.
        
        Args:
            embedding_dim: Dimension of the bottleneck embedding (default: 128)
        """
        super(GEI_Autoencoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
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
        
        # Fully connected to bottleneck
        self.enc_fc = nn.Linear(self.flatten_size, embedding_dim)
        
        # ====== DECODER ======
        # Bottleneck → Reconstruct
        
        self.dec_fc = nn.Linear(embedding_dim, self.flatten_size)
        
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
        Encode GEI image to embedding.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64)
        
        Returns:
            Embedding tensor (batch, embedding_dim)
        """
        # Encoder forward pass
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = F.relu(self.enc_bn3(self.enc_conv3(x)))
        x = F.relu(self.enc_bn4(self.enc_conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Bottleneck embedding
        embedding = self.enc_fc(x)
        
        return embedding
    
    def decode(self, embedding):
        """
        Decode embedding back to GEI image.
        
        Args:
            embedding: Embedding tensor (batch, embedding_dim)
        
        Returns:
            Reconstructed GEI tensor (batch, 1, 128, 64)
        """
        # Expand embedding
        x = self.dec_fc(embedding)
        
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
        Full autoencoder forward pass.
        
        Args:
            x: Input GEI tensor (batch, 1, 128, 64)
        
        Returns:
            reconstruction: Reconstructed GEI (batch, 1, 128, 64)
            embedding: Bottleneck embedding (batch, embedding_dim)
        """
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        
        return reconstruction, embedding


def create_autoencoder(embedding_dim=128):
    """
    Factory function to create autoencoder with weight initialization.
    
    Args:
        embedding_dim: Dimension of bottleneck embedding
    
    Returns:
        Initialized autoencoder model
    """
    model = GEI_Autoencoder(embedding_dim=embedding_dim)
    
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
    model = create_autoencoder(embedding_dim=128)
    print(model)
    
    # Test with dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 128, 64)
    
    reconstruction, embedding = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
