"""
Multi-Scale Price Transformer for financial time-series.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from neurotrader.models.blocks import (
    PatchEmbedding,
    TransformerEncoderLayer,
    CrossAttentionLayer,
)
from neurotrader.models.embeddings import (
    TimeframeEmbedding,
    InstrumentEmbedding,
    PositionalEncoding,
)
from neurotrader.models.heads import MultiTaskHead


class TimeframeEncoder(nn.Module):
    """Encoder for a single timeframe."""
    
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True,
    ):
        """
        Initialize timeframe encoder.
        
        Args:
            n_features: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            patch_len: Patch length
            stride: Patch stride
            dropout: Dropout probability
            activation: Activation function
            pre_norm: Use pre-normalization
        """
        super().__init__()
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(n_features, d_model, patch_len, stride)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, d_ff, dropout, activation, pre_norm
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, n_features]
            
        Returns:
            Encoded tensor [batch, n_patches, d_model]
        """
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        return x


class MultiScaleFusion(nn.Module):
    """Fusion module for combining multiple timeframe encodings."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        fusion_type: str = "cross_attention",
    ):
        """
        Initialize multi-scale fusion.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of fusion layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function
            fusion_type: Type of fusion (cross_attention, pooling, concat)
        """
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "cross_attention":
            self.fusion_layers = nn.ModuleList([
                CrossAttentionLayer(d_model, n_heads, d_ff, dropout, activation)
                for _ in range(n_layers)
            ])
        elif fusion_type == "pooling":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif fusion_type == "concat":
            # Will need to adjust based on number of timeframes
            pass
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        encodings: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse multiple timeframe encodings.
        
        Args:
            encodings: Dictionary of {timeframe: encoded_tensor}
            
        Returns:
            Fused representation [batch, d_model]
        """
        if self.fusion_type == "cross_attention":
            # Use first timeframe as query, attend to all others
            timeframes = list(encodings.keys())
            query = encodings[timeframes[0]]
            
            # Apply cross-attention layers
            for layer in self.fusion_layers:
                # Attend to all timeframes
                for tf in timeframes[1:]:
                    key_value = encodings[tf]
                    query = layer(query, key_value, key_value)
            
            # Pool to single vector
            fused = query.mean(dim=1)  # [batch, d_model]
            
        elif self.fusion_type == "pooling":
            # Simple mean pooling across timeframes
            all_encodings = torch.stack(list(encodings.values()), dim=1)
            # [batch, n_timeframes, n_patches, d_model]
            
            # Pool across patches
            pooled = all_encodings.mean(dim=2)  # [batch, n_timeframes, d_model]
            
            # Pool across timeframes
            fused = pooled.mean(dim=1)  # [batch, d_model]
        
        else:
            raise NotImplementedError(f"Fusion type {self.fusion_type} not implemented")
        
        return self.norm(fused)


class MultiScalePriceTransformer(nn.Module):
    """
    Multi-Scale Price Transformer for financial time-series prediction.
    
    Processes multiple timeframes independently then fuses them for prediction.
    """
    
    def __init__(
        self,
        n_features_per_tf: Dict[str, int],
        d_model: int = 256,
        n_heads: int = 8,
        n_layers_tf: int = 2,
        n_layers_fusion: int = 2,
        ff_mult: int = 4,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True,
        regression_type: str = "gaussian_nll",
        n_classes_short: int = 3,
        n_classes_long: int = 3,
        quantiles: Optional[List[float]] = None,
        use_timeframe_emb: bool = True,
        use_instrument_emb: bool = False,
        n_instruments: int = 1,
        fusion_type: str = "cross_attention",
    ):
        """
        Initialize Multi-Scale Price Transformer.
        
        Args:
            n_features_per_tf: Dictionary of {timeframe: n_features}
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers_tf: Number of layers per timeframe encoder
            n_layers_fusion: Number of fusion layers
            ff_mult: Feed-forward multiplier
            patch_len: Patch length
            stride: Patch stride
            dropout: Dropout probability
            activation: Activation function
            pre_norm: Use pre-normalization
            regression_type: Type of regression head
            n_classes_short: Number of classes for short-term trend
            n_classes_long: Number of classes for long-term trend
            quantiles: Quantiles for quantile regression
            use_timeframe_emb: Use timeframe embeddings
            use_instrument_emb: Use instrument embeddings
            n_instruments: Number of instruments
            fusion_type: Type of multi-scale fusion
        """
        super().__init__()
        
        self.timeframes = list(n_features_per_tf.keys())
        self.d_model = d_model
        self.use_timeframe_emb = use_timeframe_emb
        self.use_instrument_emb = use_instrument_emb
        
        d_ff = d_model * ff_mult
        
        # Timeframe encoders
        self.encoders = nn.ModuleDict({
            tf: TimeframeEncoder(
                n_features=n_feat,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers_tf,
                d_ff=d_ff,
                patch_len=patch_len,
                stride=stride,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm,
            )
            for tf, n_feat in n_features_per_tf.items()
        })
        
        # Timeframe embeddings
        if use_timeframe_emb:
            self.timeframe_emb = TimeframeEmbedding(len(self.timeframes), d_model)
        
        # Instrument embeddings
        if use_instrument_emb:
            self.instrument_emb = InstrumentEmbedding(n_instruments, d_model)
        
        # Multi-scale fusion
        self.fusion = MultiScaleFusion(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_fusion,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            fusion_type=fusion_type,
        )
        
        # Multi-task prediction heads
        self.heads = MultiTaskHead(
            d_model=d_model,
            regression_type=regression_type,
            n_classes_short=n_classes_short,
            n_classes_long=n_classes_long,
            quantiles=quantiles,
            dropout=dropout,
        )
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        instrument_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            inputs: Dictionary of {timeframe: input_tensor}
                   Each tensor is [batch, seq_len, n_features]
            instrument_ids: Instrument IDs [batch] (optional)
            
        Returns:
            Dictionary with predictions for each task
        """
        batch_size = next(iter(inputs.values())).size(0)
        
        # Encode each timeframe
        encodings = {}
        for tf_idx, tf in enumerate(self.timeframes):
            if tf not in inputs:
                raise ValueError(f"Missing input for timeframe {tf}")
            
            # Encode timeframe
            encoded = self.encoders[tf](inputs[tf])
            
            # Add timeframe embedding
            if self.use_timeframe_emb:
                tf_emb = self.timeframe_emb(
                    torch.tensor([tf_idx], device=encoded.device).expand(batch_size)
                )
                encoded = encoded + tf_emb.unsqueeze(1)
            
            encodings[tf] = encoded
        
        # Fuse multi-scale representations
        fused = self.fusion(encodings)
        
        # Add instrument embedding
        if self.use_instrument_emb and instrument_ids is not None:
            inst_emb = self.instrument_emb(instrument_ids)
            fused = fused + inst_emb
        
        # Generate predictions
        outputs = self.heads(fused)
        
        return outputs
