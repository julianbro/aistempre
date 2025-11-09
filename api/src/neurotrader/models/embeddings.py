"""
Embedding modules for Multi-Scale Transformer.
"""

import torch
import torch.nn as nn
import math


class TimeframeEmbedding(nn.Module):
    """Learnable embeddings for different timeframes."""
    
    def __init__(self, n_timeframes: int, d_model: int):
        """
        Initialize timeframe embedding.
        
        Args:
            n_timeframes: Number of timeframes
            d_model: Model dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(n_timeframes, d_model)
    
    def forward(self, timeframe_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            timeframe_ids: Timeframe indices [batch_size]
            
        Returns:
            Embeddings [batch_size, d_model]
        """
        return self.embedding(timeframe_ids)


class InstrumentEmbedding(nn.Module):
    """Learnable embeddings for different instruments."""
    
    def __init__(self, n_instruments: int, d_model: int):
        """
        Initialize instrument embedding.
        
        Args:
            n_instruments: Number of instruments
            d_model: Model dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(n_instruments, d_model)
    
    def forward(self, instrument_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            instrument_ids: Instrument indices [batch_size]
            
        Returns:
            Embeddings [batch_size, d_model]
        """
        return self.embedding(instrument_ids)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize learned positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for attention."""
    
    def __init__(self, d_model: int, max_relative_position: int = 128):
        """
        Initialize relative positional encoding.
        
        Args:
            d_model: Model dimension
            max_relative_position: Maximum relative distance
        """
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Embeddings for relative positions
        self.embeddings = nn.Embedding(2 * max_relative_position + 1, d_model)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get relative position embeddings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position embeddings [seq_len, seq_len, d_model]
        """
        # Create relative position matrix
        positions = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
        
        # Clip to max relative position
        positions = torch.clamp(
            positions, -self.max_relative_position, self.max_relative_position
        )
        
        # Shift to positive indices
        positions = positions + self.max_relative_position
        
        # Get embeddings
        return self.embeddings(positions.to(self.embeddings.weight.device))
