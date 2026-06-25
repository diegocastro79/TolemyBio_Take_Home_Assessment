import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    MLP Decoder with a single hidden layer.
    
    The network decodes from a high-dimensional representation to a 
    single output dimension using one hidden layer with ReLU activation.
    
    Architecture:
        Input (in_dim) -> Linear -> ReLU -> Linear -> Output (1)
    
    Args:
        in_dim (int): Input dimension
        hidden_dim (int): Hidden layer dimension
    
    Example:
        >>> decoder = Decoder(in_dim=32, hidden_dim=16)
        >>> x = torch.randn(10, 32)  # batch_size=10, in_dim=32
        >>> output = decoder(x)       # shape: (10, 1)
    """
    
    def __init__(self, in_dim: int, hidden_dim: int):
        super(Decoder, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        return self.network(x)
