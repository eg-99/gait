"""
Contrastive Loss Functions for GEI Embedding Learning

Implements InfoNCE (NT-Xent) loss for contrastive learning.
Pulls embeddings of the same person together and pushes different people apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(embeddings, labels, temperature=0.07):
    """
    InfoNCE (NT-Xent) contrastive loss.
    
    For each sample in the batch, treats all samples with the same label as positives
    and all others as negatives.
    
    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim) - normalized embeddings
        labels: Tensor of shape (batch_size,) - subject labels
        temperature: Temperature parameter for scaling (default: 0.07)
    
    Returns:
        loss: Contrastive loss scalar
    """
    batch_size = embeddings.size(0)
    device = embeddings.device
    
    # Normalize embeddings to unit sphere
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix (batch_size, batch_size)
    similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (B, B)
    
    # Create mask for positive pairs (same label)
    labels = labels.contiguous().view(-1, 1)  # (B, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)
    
    # Remove diagonal (self-similarity)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    
    # Scale by temperature
    similarity_matrix = similarity_matrix / temperature
    
    # For numerical stability, subtract max
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Compute exp(logits) for all pairs
    exp_logits = torch.exp(logits) * logits_mask
    
    # Compute log_prob = log(exp(pos) / sum(exp(all)))
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    # Compute mean log-likelihood of positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    
    # Loss is negative log-likelihood
    loss = -mean_log_prob_pos.mean()
    
    return loss


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    """
    Supervised Contrastive Loss (SupCon).
    
    Similar to InfoNCE but explicitly handles multiple positives per anchor.
    
    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim) - normalized embeddings
        labels: Tensor of shape (batch_size,) - subject labels
        temperature: Temperature parameter for scaling (default: 0.07)
    
    Returns:
        loss: Contrastive loss scalar
    """
    batch_size = embeddings.size(0)
    device = embeddings.device
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Create mask for positive pairs
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Remove diagonal
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    
    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Compute exp(logits)
    exp_logits = torch.exp(logits) * logits_mask
    
    # Compute log_prob
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    # Compute mean log-likelihood of positive pairs
    # Average over all positive pairs for each anchor
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    
    # Loss
    loss = -mean_log_prob_pos.mean()
    
    return loss


def contrastive_loss_with_hard_negatives(embeddings, labels, temperature=0.07, 
                                         hard_negative_ratio=0.5):
    """
    Contrastive loss with hard negative mining.
    
    Focuses on hard negatives (similar but different people) to improve learning.
    
    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim) - normalized embeddings
        labels: Tensor of shape (batch_size,) - subject labels
        temperature: Temperature parameter for scaling
        hard_negative_ratio: Ratio of hard negatives to use (0.0 to 1.0)
    
    Returns:
        loss: Contrastive loss scalar
    """
    batch_size = embeddings.size(0)
    device = embeddings.device
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Create mask for positive pairs
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Remove diagonal
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    negative_mask = (1 - mask) * logits_mask  # All negatives
    
    # Scale by temperature
    similarity_matrix = similarity_matrix / temperature
    
    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Hard negative mining: select top-k hardest negatives
    if hard_negative_ratio > 0:
        # Get negative similarities (higher = harder)
        negative_similarities = logits * negative_mask + (-1e9) * (1 - negative_mask)
        
        # Select top-k hardest negatives
        num_hard_negatives = int(batch_size * hard_negative_ratio)
        if num_hard_negatives > 0:
            _, hard_negative_indices = torch.topk(negative_similarities, 
                                                  k=min(num_hard_negatives, batch_size - 1),
                                                  dim=1)
            
            # Create mask for hard negatives only
            hard_negative_mask = torch.zeros_like(negative_mask)
            for i in range(batch_size):
                hard_negative_mask[i, hard_negative_indices[i]] = 1.0
            
            # Use only hard negatives in denominator
            exp_logits = torch.exp(logits) * (mask + hard_negative_mask)
        else:
            exp_logits = torch.exp(logits) * logits_mask
    else:
        exp_logits = torch.exp(logits) * logits_mask
    
    # Compute log_prob
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    # Compute mean log-likelihood of positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    
    # Loss
    loss = -mean_log_prob_pos.mean()
    
    return loss



