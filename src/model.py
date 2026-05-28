"""
Model Module
────────────
STRTransformer for WSI classification on random-walk sequences.

Two backbone options, selected via config.BACKBONE:

  "transformer" — custom nn.TransformerEncoder
      Linear proj → Sinusoidal PE → CLS prepend → TransformerEncoder
      → CLS out + mean pool → classifier

  "bert"        — HuggingFace BertModel (from scratch, no pre-trained weights)
      Linear proj → CLS prepend → BertModel(inputs_embeds)
      → CLS out + mean pool → classifier
      (BERT adds its own learned position embeddings internally)

Both produce logits of shape (B, num_classes). train.py is unchanged.
"""

import numpy as np
import torch
import torch.nn as nn

import config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).
    Used only when BACKBONE='transformer'.
    """

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


class AttentionPooling(nn.Module):
    """Gated attention pooling (ABMIL — Ilse et al., 2018).

    Learns a scalar importance weight per patch position:
        a_i = softmax( W2 · tanh(W1 · h_i) )
        out = Σ a_i · h_i

    Padding positions are masked to -inf before softmax so they
    contribute zero weight regardless of their hidden state.
    """

    def __init__(self, d_model, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, valid):
        """
        Args:
            h     : (B, L, d_model)  patch hidden states
            valid : (B, L, 1)        1.0 for real positions, 0.0 for padding
        Returns:
            out     : (B, d_model)   weighted sum
            weights : (B, L)         attention weights (for visualization)
        """
        scores = self.attn(h)                                  # (B, L, 1)
        scores = scores.masked_fill(valid == 0, float("-inf")) # mask padding
        weights = torch.softmax(scores, dim=1)                 # (B, L, 1)
        out = (weights * h).sum(dim=1)                         # (B, d_model)
        return out, weights.squeeze(-1)                        # (B, d_model), (B, L)


class STRTransformer(nn.Module):
    """
    Spatial-Temporal Random-walk Transformer for WSI multi-class classification.

    Both backbones share the same structure around the encoder:
        (B, L, in_dim)
          → Linear(in_dim, d_model)         # projection
          [→ Sinusoidal PE]                  # transformer only
          → prepend CLS token               # (B, L+1, d_model)
          → Encoder                         # (B, L+1, d_model)
          → CLS out [:, 0, :]               # (B, d_model)
          → mean pool [:, 1:, :] (masked)   # (B, d_model)
          → concat [CLS ; mean]             # (B, 2*d_model)
          → Linear(2*d_model, num_classes)  # (B, num_classes)
    """

    def __init__(
        self,
        in_dim=768,
        num_classes=None,
        d_model=None,
        nhead=None,
        num_layers=None,
        dropout=None,
        backbone=None,
    ):
        super().__init__()
        num_classes  = num_classes or config.NUM_CLASSES
        d_model      = d_model     or config.D_MODEL
        nhead        = nhead       or config.NHEAD
        num_layers   = num_layers  or config.NUM_LAYERS
        dropout      = dropout if dropout is not None else config.DROPOUT
        backbone     = backbone    or config.BACKBONE

        if backbone not in ("transformer", "bert"):
            raise ValueError(
                f"config.BACKBONE must be 'transformer' or 'bert', got '{backbone}'"
            )
        self.backbone_name = backbone

        aggregation = getattr(config, 'AGGREGATION', 'concat')
        if aggregation not in ("cls", "mean", "concat", "attention"):
            raise ValueError(
                f"config.AGGREGATION must be 'cls', 'mean', 'concat', or 'attention', "
                f"got '{aggregation}'"
            )
        self.aggregation = aggregation

        # ── Shared layers ──────────────────────────────────────────────────
        # LayerNorm sur les embeddings bruts : les encodeurs de fondation
        # (DINOv2, UNI, H-Optimus) produisent des features de magnitudes très
        # différentes. Normaliser en entrée stabilise l'entraînement et évite
        # le collapse vers une seule classe.
        self.input_norm = nn.LayerNorm(in_dim)

        # USE_NATIVE_DIM=True → Identity, travaille à in_dim (aucune compression)
        # USE_NATIVE_DIM=False → Linear(in_dim, d_model) compression vers D_MODEL
        if getattr(config, 'USE_NATIVE_DIM', False):
            d_model = in_dim
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(in_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Backbone ───────────────────────────────────────────────────────
        if backbone == "bert":
            from transformers import BertConfig, BertModel
            bert_cfg = BertConfig(
                hidden_size                  = d_model,
                num_hidden_layers            = num_layers,
                num_attention_heads          = nhead,
                intermediate_size            = 4 * d_model,
                hidden_dropout_prob          = dropout,
                attention_probs_dropout_prob = dropout,
                max_position_embeddings      = 512,
            )
            # add_pooling_layer=False : on n'utilise pas le pooler BERT,
            # on gère CLS + mean pooling nous-mêmes
            self.encoder = BertModel(bert_cfg, add_pooling_layer=False)

        else:  # "transformer"
            self.pos = PositionalEncoding(d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model        = d_model,
                nhead          = nhead,
                dim_feedforward= 4 * d_model,
                dropout        = dropout,
                batch_first    = True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ── Attention pooling (ABMIL) — uniquement si AGGREGATION="attention" ──
        if aggregation == "attention":
            attn_hidden = getattr(config, "ATTENTION_HIDDEN_DIM", 128)
            self.attn_pool = AttentionPooling(d_model, hidden_dim=attn_hidden)

        # ── Classification head ────────────────────────────────────────────
        # head_dim dépend de AGGREGATION :
        #   "cls" / "mean" / "attention" → d_model   |   "concat" → 2 * d_model
        head_dim = 2 * d_model if aggregation == "concat" else d_model
        self.cls_head = nn.Linear(head_dim, num_classes)

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x, lengths):
        """
        Args:
            x       : (B, L, in_dim)  padded patch embeddings
            lengths : (B,)            real walk lengths (before padding)
        Returns:
            logits  : (B, num_classes)
        """
        B, L, _ = x.shape

        # 1. Normalize raw embeddings, then project to d_model  →  (B, L, d_model)
        x = self.input_norm(x)
        x = self.proj(x)

        # 2. Sinusoidal positional encoding (transformer only)
        #    BERT adds its own learned position embeddings internally
        if self.backbone_name == "transformer":
            x = self.pos(x)

        # 3. Prepend CLS token  →  (B, L+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        # 4. Encode ────────────────────────────────────────────────────────
        if self.backbone_name == "bert":
            # BERT convention: 1 = valid, 0 = padded  (opposite of PyTorch)
            # CLS is always valid (position 0 → always < lengths+1)
            attn_mask = (
                torch.arange(L + 1, device=x.device)
                .unsqueeze(0)
                .expand(B, L + 1)
                < (lengths + 1).unsqueeze(1)
            ).long()                                      # (B, L+1)

            h = self.encoder(
                inputs_embeds  = x,
                attention_mask = attn_mask,
            ).last_hidden_state                           # (B, L+1, d_model)

            valid = attn_mask[:, 1:].unsqueeze(-1).float()  # (B, L, 1)

        else:  # "transformer"
            # PyTorch convention: True = padded (ignored)
            pad_mask = (
                torch.arange(L + 1, device=x.device)
                .unsqueeze(0)
                .expand(B, L + 1)
                >= (lengths + 1).unsqueeze(1)
            )                                             # (B, L+1)

            h = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, L+1, d_model)

            valid = (~pad_mask[:, 1:]).unsqueeze(-1).float()     # (B, L, 1)

        # 5. CLS token output  →  (B, d_model)
        cls_out = h[:, 0, :]

        # 6. Mean pooling over patch positions (masked)  →  (B, d_model)
        patch_h  = h[:, 1:, :]
        mean_out = (patch_h * valid).sum(1) / valid.sum(1).clamp_min(1.0)

        # 7. Agréger selon AGGREGATION + classifier  →  (B, num_classes)
        if self.aggregation == "cls":
            out = cls_out
        elif self.aggregation == "mean":
            out = mean_out
        elif self.aggregation == "attention":
            out, _ = self.attn_pool(patch_h, valid)
        else:  # "concat"
            out = torch.cat([cls_out, mean_out], dim=1)
        return self.cls_head(out)
