"""
Visual encoder for pixel observations: [B, C, H, W] → [B, d].

Three stride-2 conv layers reduce a (img_size × img_size) frame to an
8×8 feature map, which is flattened and projected linearly to the latent
dimension d.  Same orthogonal init on the final linear as the MLP encoder.
"""

import math
import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """
    CNN front-end for pixel observations.

    Architecture (default 64×64 input):
        Conv(C → 32, 4×4, stride 2, pad 1) → ReLU   [B, 32, 32, 32]
        Conv(32 → 64, 4×4, stride 2, pad 1) → ReLU  [B, 64, 16, 16]
        Conv(64 → 128, 4×4, stride 2, pad 1) → ReLU [B, 128, 8, 8]
        Flatten → Linear(128 * h_out * w_out, d)

    h_out = w_out = img_size // 8  (three stride-2 halvings).
    """

    def __init__(self, img_channels: int = 3, img_size: int = 64, d: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32,  4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32,           64,  4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,           128, 4, stride=2, padding=1), nn.ReLU(),
        )
        # After 3 stride-2 halvings: spatial size = img_size // 8
        h_out = img_size // 8
        flat_dim = 128 * h_out * h_out
        self.proj = nn.Linear(flat_dim, d)
        nn.init.orthogonal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], pixel values in [0, 1]
        h = self.cnn(x)
        return self.proj(h.flatten(1))
