import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    MLP Decoder with a single hidden layer.
    
    The network decodes from the embedding representation back to the gene expression vector

    Args:
        in_dim (int): Input dimension (embedding dimension of the encoder)
        hidden_dim (int): Hidden layer dimension
        out_dim: original gene expression dimension
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus(), # to make sure a positive outcome
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_cells, in_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_cells, out_dim)
        """
        return self.network(x)


class ProbLayer(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)

