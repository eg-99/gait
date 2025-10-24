"""
GEI-based CNN Model for Gait Recognition

This model uses Gait Energy Images (GEI) - averaged silhouettes that capture
the temporal gait pattern in a single image. A simple CNN is used for classification.

Architecture:
- Input: GEI image (1 x 128 x 64)
- Conv layers with batch normalization and dropout
- Fully connected layers
- Output: Subject classification (124 classes)

Expected Performance: 70-85% accuracy on CASIA-B dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GEI_CNN(nn.Module):
    """
    Convolutional Neural Network for GEI-based gait recognition.
    
    This is a relatively simple but effective baseline model.
    """
    def __init__(self, num_classes: int = 124, dropout: float = 0.5):
        """
        Initialize the GEI CNN model.
        
        Args:
            num_classes: Number of subjects to classify
            dropout: Dropout probability for regularization
        """
        super(GEI_CNN, self).__init__()
        
        # Convolutional Block 1: Extract low-level features
        # Input: (batch, 1, 128, 64)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> (batch, 32, 64, 32)
        
        # Convolutional Block 2: Extract mid-level features
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> (batch, 64, 32, 16)
        
        # Convolutional Block 3: Extract high-level features
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> (batch, 128, 16, 8)
        
        # Convolutional Block 4: Deep features
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> (batch, 256, 8, 4)
        
        # Calculate the size after convolutions: 256 * 8 * 4 = 8192
        self.fc_input_size = 256 * 8 * 4
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 1, 128, 64)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Convolutional layers with ReLU activations and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.dropout1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn_fc2(self.fc2(x))))
        
        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x
    
    def extract_features(self, x):
        """
        Extract feature representation before the final classification layer.
        Useful for visualization or transfer learning.
        
        Args:
            x: Input tensor of shape (batch, 1, 128, 64)
        
        Returns:
            Feature tensor of shape (batch, 256)
        """
        # Same as forward but stop before final layer
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn_fc2(self.fc2(x))))
        
        return x


def create_gei_cnn(num_classes: int = 124, dropout: float = 0.5, pretrained: bool = False) -> GEI_CNN:
    """
    Create a GEI CNN model.
    
    Args:
        num_classes: Number of subjects to classify
        dropout: Dropout probability
        pretrained: If True, load pretrained weights (not implemented yet)
    
    Returns:
        GEI_CNN model
    """
    model = GEI_CNN(num_classes=num_classes, dropout=dropout)
    
    # Initialize weights using Kaiming initialization for ReLU activation
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_gei_cnn(num_classes=124)
    print(model)
    
    # Test with dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 128, 64)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
