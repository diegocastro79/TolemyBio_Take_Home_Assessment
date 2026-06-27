import torch
import torch.nn as nn
from typing import List


class MLPEncoder(nn.Module):
    """
    Multi-Layer Perceptron Encoder with configurable hidden layers.
    
    The network has a fixed input dimension of 1 and customizable hidden layer
    dimensions with ReLU activations and dropout regularization.
    
    Architecture:
        Input (1) -> Linear -> ReLU -> Dropout -> ... -> Linear (output)
    
    Args:
        in_dim: input dimension (number of genes)
        hidden_dim: embedding dimension
        n_layers: number of hidden layers
        dropout_rate (float, optional): Dropout probability applied after each 
                                       hidden layer. Defaults to 0.1.
    
    Example:
        >>> encoder = MLPEncoder(in_dim=2000, hidden_dim=100, n_layers=4, dropout_rate=0.2)
        >>> x = torch.randn(10, 2000)  # batch_size=10, input_dim=2000
        >>> output = encoder(x)     # shape: (10, 32)
    """
    
    def __init__(self, in_dim:int, hidden_dim: int, emb_dim: int, n_layers: int, dropout_rate: float = 0.1):
        super(MLPEncoder, self).__init__()
        
        layers = []
        dims = [in_dim] + [hidden_dim] * n_layers + [emb_dim]

        for i, dim in enumerate(dims[:-1]):
            if i < len(dims) - 2:
                layers += [nn.Linear(dim, dims[i+1]), nn.GELU(), nn.Dropout(p=dropout_rate)]
            else:
                layers += [nn.Linear(dim, dims[i+1])]

        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_cells, num_gens)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_cells, hidden_dim)
        """
        return self.network(x)
