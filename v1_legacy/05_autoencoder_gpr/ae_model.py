#!/usr/bin/env python3
"""
Improved Autoencoder Architecture with Monotonicity Constraint
===============================================================
This version includes:
- Increased latent dimensions for damage
- Smoothness regularization
- Monotonicity-aware training
- Better architecture with skip connections
"""

import torch
import torch.nn as nn


class ImprovedCurveAutoencoder(nn.Module):
    """
    Enhanced autoencoder with:
    - Deeper architecture
    - Dropout for regularization
    - Optional skip connections
    """
    def __init__(self, n_points: int, latent_dim: int = 16, use_skip: bool = False):
        super().__init__()

        self.n_points = n_points
        self.latent_dim = latent_dim
        self.use_skip = use_skip

        # Encoder with gradual compression
        self.encoder = nn.Sequential(
            nn.Linear(n_points, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, latent_dim),
        )

        # Decoder with gradual expansion
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            
            nn.Linear(128, n_points),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


class MonotonicDamageAutoencoder(nn.Module):
    """
    Autoencoder specifically designed for damage curves.
    Enforces monotonicity through cumulative sum in decoder.
    """
    def __init__(self, n_points: int, latent_dim: int = 12):
        super().__init__()

        self.n_points = n_points
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_points, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, latent_dim),
        )

        # Decoder outputs increments (not absolute values)
        self.decoder_increments = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            
            nn.Linear(128, n_points),
            nn.Softplus(),  # Ensures positive increments
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        # Decode to increments
        increments = self.decoder_increments(z)
        
        # Apply cumulative sum to ensure monotonicity
        damage = torch.cumsum(increments, dim=-1)
        
        # Normalize to [0, 1] range
        damage_max = damage[:, -1:].clamp(min=1e-6)
        damage = damage / damage_max
        
        return damage

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


# Smooth L1 loss for better robustness
class SmoothL1ReconstructionLoss(nn.Module):
    """Combined reconstruction + smoothness loss"""
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(self, pred, target):
        # Reconstruction loss
        recon_loss = self.smooth_l1(pred, target)
        
        # Smoothness loss (penalize large differences between adjacent points)
        diff = pred[:, 1:] - pred[:, :-1]
        smoothness_loss = torch.mean(diff ** 2)
        
        return recon_loss + self.alpha * smoothness_loss