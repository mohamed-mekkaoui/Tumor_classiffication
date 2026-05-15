"""
Model Module
────────────
Spatial-Temporal Random-walk Transformer (STRTransformer) for multi-class
WSI classification.  Takes a sequence of patch embeddings (one random walk)
and outputs class logits.
"""

import numpy as np
import torch
import torch.nn as nn

import config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class STRTransformer(nn.Module):
    """Transformer encoder that classifies a variable-length sequence of
    patch embeddings into one of ``num_classes`` categories.

    Pipeline:
        input (B, L, in_dim)
          → Linear projection (in_dim → d_model)
          → Positional Encoding
          → TransformerEncoder (num_layers × self-attention)
          → Mean pooling over valid (non-padded) positions
          → Linear classifier → logits (B, num_classes)
    """

    def __init__(
        self,
        in_dim=768,
        num_classes=None,
        d_model=None,
        nhead=None,
        num_layers=None,
        dropout=None,
    ):
        super().__init__()
        num_classes = num_classes or config.NUM_CLASSES
        d_model = d_model or config.D_MODEL
        nhead = nhead or config.NHEAD
        num_layers = num_layers or config.NUM_LAYERS
        dropout = dropout if dropout is not None else config.DROPOUT

        self.proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        """
        Args:
            x       : (B, L, in_dim)  padded input sequences
            lengths : (B,)            actual lengths before padding

        Returns:
            logits  : (B, num_classes)
        """
        x = self.proj(x)
        x = self.pos(x)

        B, L, _ = x.shape
        # True where position is padding → masked out by TransformerEncoder
        pad_mask = (
            torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            >= lengths.unsqueeze(1)
        )

        h = self.enc(x, src_key_padding_mask=pad_mask)  # (B, L, d_model)

        # Mean-pool over valid positions only
        valid = (~pad_mask).unsqueeze(-1).float()  # (B, L, 1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        return self.cls(pooled)
