"""
paft_core.py
=======================================================================
Polar-Adapted Fine-Tuning (PAFT) Core Modules
Replaces standard GPT-2 attention linear layers with PAFT layers.
Supports both 'Pure' and 'Hybrid' training modes.
=======================================================================
"""

import torch
import torch.nn as nn
from scipy.linalg import polar
import numpy as np


def get_polar_components(W_numpy):
    """
    Computes Polar Decomposition (W = Q * S).
    Returns Q (Isometry) and S (Symmetric Stretch) as PyTorch tensors.
    """
    # Ensure float64 for stable polar decomposition
    Q, S = polar(W_numpy.astype(np.float64))
    return torch.tensor(Q, dtype=torch.float32), torch.tensor(S, dtype=torch.float32)


class PAFT_Attention_Wv(nn.Module):
    """
    Replaces GPT-2's fused c_attn block.
    Freezes W_q and W_k. Applies PAFT (Freeze Q, Train S) strictly to W_v.
    """

    def __init__(self, original_c_attn):
        super().__init__()

        # 1. Extract weights from the original pre-trained layer
        W = original_c_attn.weight.data.cpu().numpy()  # Shape: [768, 2304]
        b = original_c_attn.bias.data  # Shape: [2304]
        d_model = W.shape[0]  # 768

        # 2. Slice into Q, K, and V
        W_q = torch.tensor(W[:, :d_model])
        W_k = torch.tensor(W[:, d_model:2 * d_model])
        W_v_numpy = W[:, 2 * d_model:]

        # 3. Apply Polar Decomposition to W_v
        Q_v, S_v = get_polar_components(W_v_numpy)

        # 4. Register parameters according to PAFT constraints
        self.register_buffer('W_q', W_q)
        self.register_buffer('W_k', W_k)
        self.register_buffer('Q_v', Q_v)  # Routing isometry is locked
        # S_v is intentionally unconstrained (not forced symmetric).
        # Per Section 2.3 "Elastic Algebra": any general S_train = Q_δ · S_new,
        # so W_final = (Q_frozen · Q_δ) · S_new — a learned residual micro-rotation
        # plus pure stretch. This provides empirical superiority over strict symmetry.
        self.S_v = nn.Parameter(S_v)  # Semantic stretch is trainable
        self.bias = nn.Parameter(b)

    def forward(self, x):
        # Dynamically reconstruct W_v = Q_v @ S_v
        W_v_reconstructed = torch.matmul(self.Q_v, self.S_v)

        # Re-concatenate the fused matrix
        W_fused = torch.cat([self.W_q, self.W_k, W_v_reconstructed], dim=1)
        return torch.matmul(x, W_fused) + self.bias


class PAFT_Output_Wo(nn.Module):
    """
    Replaces GPT-2's c_proj block (W_o).
    Applies PAFT (Freeze Q, Train S) to the output projection.
    """

    def __init__(self, original_c_proj):
        super().__init__()

        W_o_numpy = original_c_proj.weight.data.cpu().numpy()
        b = original_c_proj.bias.data

        # Apply Polar Decomposition to W_o
        Q_o, S_o = get_polar_components(W_o_numpy)

        self.register_buffer('Q_o', Q_o)  # Routing isometry is locked
        self.S_o = nn.Parameter(S_o)  # Semantic stretch is trainable
        self.bias = nn.Parameter(b)

    def forward(self, x):
        # Reconstruct W_o = Q_o @ S_o
        W_o_reconstructed = torch.matmul(self.Q_o, self.S_o)
        return torch.matmul(x, W_o_reconstructed) + self.bias


def convert_model_to_paft(model, mode="pure"):
    """
    Replaces standard matrices with PAFT layers and toggles gradients.

    Args:
        model: The GPT-2 model instance.
        mode (str): "pure" (S-only) or "hybrid" (S + MLPs + LayerNorms).
    """
    print(f"\n  [PAFT] Converting model to {mode.upper()} mode...")

    for i in range(len(model.transformer.h)):
        # 1. Replace W_v circuit
        orig_c_attn = model.transformer.h[i].attn.c_attn
        model.transformer.h[i].attn.c_attn = PAFT_Attention_Wv(orig_c_attn)

        # 2. Replace W_o circuit
        orig_c_proj = model.transformer.h[i].attn.c_proj
        model.transformer.h[i].attn.c_proj = PAFT_Output_Wo(orig_c_proj)

    # Gradient Locking Logic
    for name, param in model.named_parameters():
        # Core PAFT components are ALWAYS trainable
        if ".S_v" in name or ".S_o" in name or ".bias" in name:
            param.requires_grad = True

        # Hybrid components are only trainable in hybrid mode
        elif mode == "hybrid" and ("mlp" in name or "ln_" in name):
            param.requires_grad = True

        # Everything else (Embeddings, Frozen Qs, Wq, Wk) is locked
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [PAFT] Complete. Mode: {mode.upper()} | Trainable Parameters: {trainable:,}")
    return model