"""
Transformer building blocks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert time series to patches and embed them."""

    def __init__(
        self,
        n_features: int,
        d_model: int,
        patch_len: int = 16,
        stride: int = 8,
    ):
        """
        Initialize patch embedding.

        Args:
            n_features: Number of input features
            d_model: Model dimension
            patch_len: Length of each patch
            stride: Stride between patches
        """
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Linear projection for patches
        self.projection = nn.Linear(n_features * patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to patches.

        Args:
            x: Input tensor [batch_size, seq_len, n_features]

        Returns:
            Patched tensor [batch_size, n_patches, d_model]
        """
        batch_size, seq_len, n_features = x.shape

        # Calculate number of patches
        n_patches = (seq_len - self.patch_len) // self.stride + 1

        # Extract patches
        patches = []
        for i in range(n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :]  # [batch, patch_len, features]
            patches.append(patch.reshape(batch_size, -1))  # Flatten patch

        patches = torch.stack(patches, dim=1)  # [batch, n_patches, patch_len*features]

        # Project patches
        return self.projection(patches)


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Use bias in projections
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model]
            value: Value tensor [batch, seq_len_v, d_model]
            mask: Attention mask [batch, seq_len_q, seq_len_k]

        Returns:
            Output tensor [batch, seq_len_q, d_model]
        """
        batch_size = query.size(0)

        # Project and reshape to multi-head
        Q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True,
    ):
        """
        Initialize transformer encoder layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
            pre_norm: Use pre-norm (True) or post-norm (False)
        """
        super().__init__()
        self.pre_norm = pre_norm

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        if self.pre_norm:
            # Pre-norm
            residual = x
            x = self.norm1(x)
            x = self.self_attn(x, x, x, mask)
            x = residual + self.dropout1(x)

            residual = x
            x = self.norm2(x)
            x = self.feed_forward(x)
            x = residual + self.dropout2(x)
        else:
            # Post-norm
            residual = x
            x = self.self_attn(x, x, x, mask)
            x = self.norm1(residual + self.dropout1(x))

            residual = x
            x = self.feed_forward(x)
            x = self.norm2(residual + self.dropout2(x))

        return x


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for fusing different timeframes."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize cross-attention layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query: Query tensor (target timeframe)
            key: Key tensor (source timeframe)
            value: Value tensor (source timeframe)

        Returns:
            Output tensor
        """
        # Cross-attention
        residual = query
        query = self.norm1(query)
        attn_output = self.cross_attn(query, key, value)
        query = residual + self.dropout1(attn_output)

        # Feed-forward
        residual = query
        query = self.norm2(query)
        ff_output = self.feed_forward(query)
        query = residual + self.dropout2(ff_output)

        return query
