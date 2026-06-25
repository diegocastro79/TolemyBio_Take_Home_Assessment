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
        dims (List[int]): List of integers specifying the width of each hidden 
                         layer and the output layer. For example, [64, 128, 32] 
                         creates hidden layers of width 64 and 128, with an 
                         output layer of width 32.
        dropout_rate (float, optional): Dropout probability applied after each 
                                       hidden layer. Defaults to 0.1.
    
    Example:
        >>> encoder = MLPEncoder(dims=[64, 128, 32], dropout_rate=0.2)
        >>> x = torch.randn(10, 1)  # batch_size=10, input_dim=1
        >>> output = encoder(x)     # shape: (10, 32)
    """
    
    def __init__(self, dims: List[int], dropout_rate: float = 0.1):
        super(MLPEncoder, self).__init__()
        
        if not dims:
            raise ValueError("dims must contain at least one element (output dimension)")
        
        self.dims = dims
        self.dropout_rate = dropout_rate
        
        layers = []
        input_dim = 1
        
        for i, hidden_dim in enumerate(dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if i < len(dims) - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_rate))
            
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dims[-1])
        """
        return self.network(x)
