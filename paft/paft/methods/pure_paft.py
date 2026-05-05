"""
PAFTAttention and PAFTModel.

PAFTAttention replaces GPT2Attention in each transformer block.
It stores:
    Frozen buffers  — Q_V, Q_O (rotation), W_Q, W_K, all biases, causal mask
    nn.Parameters   — S_V, S_O (scaling)  [trainable by default; frozen by method]

PAFTModel wraps GPT2LMHeadModel, replacing every attention layer with
PAFTAttention at construction time.  All HuggingFace utilities (generate,
save_pretrained, etc.) work via the inner self.base model.

Weight shapes (GPT-2 small: n_embd=768, n_heads=12, d_head=64):
    W_V_h = Q_V[h] @ S_V[h]    [768,64] = [768,64] @ [64,64]
    W_O_h = S_O[h] @ Q_O[h]    [64,768] = [64,64]  @ [64,768]

    Stacked:
        W_V_full = cat(W_V_h, dim=1)  [768, 768]
        W_O_full = cat(W_O_h, dim=0)  [768, 768]   (rows from each head)

Forward pass reconstructs W_V_full and W_O_full on every call so gradients
flow through S_V and S_O.  W_Q, W_K, and all biases use frozen buffers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from paft.decomposition.polar import polar_decompose, polar_decompose_left
from paft.model.extractor import (
    get_gpt2_dims,
    extract_wv_head,
    extract_wo_head,
    extract_wq_wk_layer,
    extract_biases_layer,
)


# ---------------------------------------------------------------------------
# PAFTAttention
# ---------------------------------------------------------------------------

class PAFTAttention(nn.Module):
    """
    Decomposed attention module that replaces GPT2Attention.

    Frozen buffers (never updated by optimizer):
        Q_V     [n_heads, n_embd, d_head]   — rotation for V projection
        Q_O     [n_heads, d_head, n_embd]   — rotation for O projection
        W_Q     [n_embd, n_embd]            — Q projection (full layer)
        W_K     [n_embd, n_embd]            — K projection (full layer)
        b_qkv   [3*n_embd]                  — c_attn bias
        b_o     [n_embd]                    — c_proj bias
        causal_mask  [1,1,max_pos,max_pos]  — lower-triangular bool

    Trainable nn.Parameters (which ones are active depends on method):
        S_V     [n_heads, d_head, d_head]   — scaling for V projection
        S_O     [n_heads, d_head, d_head]   — scaling for O projection
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        Q_V:        torch.Tensor,   # [n_heads, n_embd, d_head]
        S_V_init:   torch.Tensor,   # [n_heads, d_head, d_head]
        Q_O:        torch.Tensor,   # [n_heads, d_head, n_embd]
        S_O_init:   torch.Tensor,   # [n_heads, d_head, d_head]
        W_Q:        torch.Tensor,   # [n_embd, n_embd]
        W_K:        torch.Tensor,   # [n_embd, n_embd]
        b_qkv:      torch.Tensor,   # [3*n_embd]
        b_o:        torch.Tensor,   # [n_embd]
    ) -> None:
        super().__init__()

        self.n_heads = config.n_head
        self.n_embd  = config.n_embd
        self.d_head  = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        self.scale   = 1.0 / math.sqrt(self.d_head)

        # --- frozen rotation / projection buffers ---
        self.register_buffer("Q_V",  Q_V)    # [n_heads, n_embd, d_head]
        self.register_buffer("Q_O",  Q_O)    # [n_heads, d_head, n_embd]
        self.register_buffer("W_Q",  W_Q)    # [n_embd, n_embd]
        self.register_buffer("W_K",  W_K)    # [n_embd, n_embd]
        self.register_buffer("b_qkv", b_qkv) # [3*n_embd]
        self.register_buffer("b_o",   b_o)   # [n_embd]

        # Causal mask — same logic as GPT2Attention
        max_pos = config.max_position_embeddings  # 1024 for all GPT-2 sizes
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_pos, max_pos, dtype=torch.bool)).view(
                1, 1, max_pos, max_pos
            ),
        )

        # --- trainable scaling matrices ---
        # Initialized from polar decomposition of the pretrained weights.
        # Both start as symmetric PSD; no symmetry constraint is enforced
        # during training (the method decides whether to add regularization).
        self.S_V = nn.Parameter(S_V_init.clone())  # [n_heads, d_head, d_head]
        self.S_O = nn.Parameter(S_O_init.clone())  # [n_heads, d_head, d_head]

    # ------------------------------------------------------------------
    # Weight reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_W_V(self) -> torch.Tensor:
        """
        Reconstruct full V-projection weight.

        W_V_h = Q_V[h] @ S_V[h]  for each head h.
        Concatenated along dim=1  →  [n_embd, n_embd].

        Gradient flows through S_V; Q_V is a frozen buffer.
        """
        cols = [
            self.Q_V[h] @ self.S_V[h]          # [n_embd, d_head]
            for h in range(self.n_heads)
        ]
        return torch.cat(cols, dim=1)            # [n_embd, n_embd]

    def _reconstruct_W_O(self) -> torch.Tensor:
        """
        Reconstruct full O-projection weight.

        W_O_h = S_O[h] @ Q_O[h]  for each head h.
        Concatenated along dim=0  →  [n_embd, n_embd].

        Gradient flows through S_O; Q_O is a frozen buffer.
        """
        rows = [
            self.S_O[h] @ self.Q_O[h]           # [d_head, n_embd]
            for h in range(self.n_heads)
        ]
        return torch.cat(rows, dim=0)            # [n_embd, n_embd]

    # ------------------------------------------------------------------
    # Head splitting / merging  (identical to GPT2Attention)
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, n_embd]  ->  [batch, n_heads, seq, d_head]"""
        b, s, _ = x.shape
        x = x.view(b, s, self.n_heads, self.d_head)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, n_heads, seq, d_head]  ->  [batch, seq, n_embd]"""
        b, _, s, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(b, s, self.n_embd)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states:  torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past:     Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask:      Optional[torch.Tensor] = None,
        use_cache:      bool = False,
        output_attentions: bool = False,
    ) -> Tuple:
        """
        Forward pass matching GPT2Attention's output signature.

        Returns (attn_output, present [, attn_weights])
            attn_output : [batch, seq, n_embd]
            present     : (k, v) tuple if use_cache else None
            attn_weights: [batch, n_heads, seq_q, seq_k] if output_attentions
        """
        # --- reconstruct decomposed weights once per forward ---
        W_V = self._reconstruct_W_V()   # [n_embd, n_embd]  — grad via S_V
        W_O = self._reconstruct_W_O()   # [n_embd, n_embd]  — grad via S_O

        b, seq, _ = hidden_states.shape

        # --- QKV projections ---
        # b_qkv is split into b_q, b_k, b_v for clarity
        b_q = self.b_qkv[:self.n_embd]
        b_k = self.b_qkv[self.n_embd : 2 * self.n_embd]
        b_v = self.b_qkv[2 * self.n_embd:]

        q = hidden_states @ self.W_Q + b_q   # [batch, seq, n_embd]  frozen W_Q
        k = hidden_states @ self.W_K + b_k   # [batch, seq, n_embd]  frozen W_K
        v = hidden_states @ W_V  + b_v       # [batch, seq, n_embd]  trainable W_V

        # --- split into heads ---
        q = self._split_heads(q)   # [batch, n_heads, seq, d_head]
        k = self._split_heads(k)
        v = self._split_heads(v)

        # --- KV cache (used during generation) ---
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present = (k, v) if use_cache else None

        # --- scaled dot-product attention ---
        kv_len = k.shape[-2]
        q_len  = q.shape[-2]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # shape: [batch, n_heads, q_len, kv_len]

        # Causal mask — prevent attending to future tokens
        mask = self.causal_mask[:, :, kv_len - q_len : kv_len, :kv_len]
        # Fill masked positions with a large negative value before softmax
        fill = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~mask, fill)

        # Optional padding mask from the data collator (additive, pre-softmax)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(v.dtype)   # fp16 safety

        # Optional per-head mask (used by HuggingFace trainer)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # --- attended values ---
        context = torch.matmul(attn_weights, v)   # [batch, n_heads, seq, d_head]

        # --- merge heads and output projection ---
        context = self._merge_heads(context)       # [batch, seq, n_embd]
        attn_output = context @ W_O + self.b_o     # [batch, seq, n_embd] trainable W_O

        outputs = (attn_output, present)
        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


# ---------------------------------------------------------------------------
# PAFTModel — wraps GPT2LMHeadModel, replaces attention layers
# ---------------------------------------------------------------------------

class PAFTModel(nn.Module):
    """
    Wraps a GPT2LMHeadModel and replaces every GPT2Attention with a
    PAFTAttention at construction time.

    Usage:
        from transformers import GPT2LMHeadModel
        base = GPT2LMHeadModel.from_pretrained('gpt2')
        model = PAFTModel(base)
        # now apply a PAFT method to set requires_grad correctly:
        PurePAFT().apply(model)

    The inner base model is accessible as model.base, so HuggingFace
    generate(), save_pretrained(), etc. all still work via model.base.
    """

    def __init__(self, base_model) -> None:
        super().__init__()
        self.base = base_model
        self._decompose_and_replace_attention()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _decompose_and_replace_attention(self) -> None:
        """Decompose all attention layers and replace in-place."""
        dims = get_gpt2_dims(self.base)
        for layer_idx in range(dims["n_layers"]):
            paft_attn = self._build_paft_attention(layer_idx, dims)
            self.base.transformer.h[layer_idx].attn = paft_attn

    def _build_paft_attention(self, layer_idx: int, dims: dict) -> PAFTAttention:
        """
        For one layer: extract weights, polar-decompose per head, build module.
        """
        n_heads = dims["n_heads"]
        cfg     = self.base.config

        Q_V_list, S_V_list = [], []
        Q_O_list, S_O_list = [], []

        for h in range(n_heads):
            # W_V_h [n_embd, d_head]  — tall  — right polar
            W_V_h = extract_wv_head(self.base, layer_idx, h).float()
            Q_V_h, S_V_h = polar_decompose(W_V_h)
            Q_V_list.append(Q_V_h)
            S_V_list.append(S_V_h)

            # W_O_h [d_head, n_embd]  — wide  — left polar
            W_O_h = extract_wo_head(self.base, layer_idx, h).float()
            S_O_h, Q_O_h = polar_decompose_left(W_O_h)
            Q_O_list.append(Q_O_h)
            S_O_list.append(S_O_h)

        Q_V = torch.stack(Q_V_list)    # [n_heads, n_embd, d_head]
        S_V = torch.stack(S_V_list)    # [n_heads, d_head, d_head]
        Q_O = torch.stack(Q_O_list)    # [n_heads, d_head, n_embd]
        S_O = torch.stack(S_O_list)    # [n_heads, d_head, d_head]

        W_Q, W_K = extract_wq_wk_layer(self.base, layer_idx)
        b_qkv, b_o = extract_biases_layer(self.base, layer_idx)

        return PAFTAttention(
            config    = cfg,
            layer_idx = layer_idx,
            Q_V       = Q_V,
            S_V_init  = S_V,
            Q_O       = Q_O,
            S_O_init  = S_O,
            W_Q       = W_Q.float(),
            W_K       = W_K.float(),
            b_qkv     = b_qkv.float(),
            b_o       = b_o.float(),
        )

    # ------------------------------------------------------------------
    # Convenience accessors (used by checkpointing and analysis)
    # ------------------------------------------------------------------

    def get_paft_attention(self, layer_idx: int) -> PAFTAttention:
        """Return the PAFTAttention module for a given layer."""
        attn = self.base.transformer.h[layer_idx].attn
        assert isinstance(attn, PAFTAttention), (
            f"Layer {layer_idx} attention is not a PAFTAttention — "
            "was PAFTModel constructed correctly?"
        )
        return attn

    def iter_paft_attentions(self):
        """Yield (layer_idx, PAFTAttention) for all layers."""
        n_layers = self.base.config.n_layer
        for i in range(n_layers):
            yield i, self.get_paft_attention(i)

    def get_all_S(self) -> dict:
        """
        Return all S_V and S_O tensors as a flat dict.

        Keys: 'S_V_{layer}' and 'S_O_{layer}'.
        Used by the checkpointing saver.
        """
        out = {}
        for i, attn in self.iter_paft_attentions():
            out[f"S_V_{i}"] = attn.S_V.detach()
            out[f"S_O_{i}"] = attn.S_O.detach()
        return out

    def get_all_Q(self) -> dict:
        """
        Return all frozen Q_V and Q_O tensors.

        Keys: 'Q_V_{layer}' and 'Q_O_{layer}'.
        Used by the checkpointing saver (saved once at init).
        """
        out = {}
        for i, attn in self.iter_paft_attentions():
            out[f"Q_V_{i}"] = attn.Q_V
            out[f"Q_O_{i}"] = attn.Q_O
        return out

    # ------------------------------------------------------------------
    # Forward — delegates entirely to base model
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        """Delegate to GPT2LMHeadModel.forward."""
        return self.base(*args, **kwargs)

    # ------------------------------------------------------------------
    # Device / dtype passthrough
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        self.base = self.base.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    # Convenience shims so callers can treat PAFTModel like GPT2LMHeadModel
    @property
    def config(self):
        return self.base.config

    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.base.save_pretrained(*args, **kwargs)