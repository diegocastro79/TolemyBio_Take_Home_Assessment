import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Head(nn.Module):
    """
    Single attention head with graph structure awareness.
    
    Implements scaled dot-product attention with masking for directed graphs.
    
    Args:
        input_dim (int): Input dimension (n_heads * head_size)
        head_size (int): Hidden dimension for this attention head
    """
    
    def __init__(self, input_dim: int, head_size: int):
        super(Head, self).__init__()
        self.head_size = head_size
        
        # Initialize Q, K, V projections
        self.query = nn.Linear(input_dim, head_size, bias=False)
        self.key = nn.Linear(input_dim, head_size, bias=False)
        self.value = nn.Linear(input_dim, head_size, bias=False)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention masking.
        
        Args:
            x (torch.Tensor): Input of shape (B, G+T, input_dim)
            mask (torch.Tensor): Attention mask of shape (G+T, G+T)
                                Float weights for edges, -inf for no edges
        
        Returns:
            torch.Tensor: Output of shape (B, G+T, head_size)
        """
        # Compute Q, K, V
        Q = self.query(x)  # (B, N, head_size)
        K = self.key(x)    # (B, N, head_size)
        V = self.value(x)  # (B, N, head_size)
        
        # Use PyTorch's efficient scaled_dot_product_attention
        # mask (N, N) broadcasts automatically to (B, N, N) for attention scores
        out = F.scaled_dot_product_attention(
            Q,  # (B, N, head_size)
            K,  # (B, N, head_size)
            V,  # (B, N, head_size)
            attn_mask=mask  # (N, N)
        )
        
        return out  # (B, N, head_size)


class MultiHead(nn.Module):
    """
    Multi-head attention mechanism.
    
    Concatenates outputs from multiple attention heads and applies projection.
    
    Args:
        input_dim (int): Input dimension (n_heads * head_size)
        head_size (int): Hidden dimension per attention head
        n_heads (int): Number of attention heads
    """
    
    def __init__(self, input_dim: int, head_size: int, n_heads: int):
        super(MultiHead, self).__init__()
        self.heads = nn.ModuleList([Head(input_dim, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_heads * head_size)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multiple attention heads.
        
        Args:
            x (torch.Tensor): Input of shape (B, G+T, input_dim)
            mask (torch.Tensor): Attention mask of shape (G+T, G+T)
        
        Returns:
            torch.Tensor: Output of shape (B, G+T, n_heads*head_size)
        """
        # Apply each head and concatenate
        out = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        
        # Apply projection
        out = self.proj(out)
        
        return out


class FFLayer(nn.Module):
    """
    Feed-forward layer with single hidden layer.
    
    Input and output dimensions are the same (dim).
    Hidden dimension is 5*dim with ReLU activation.
    
    Args:
        dim (int): Input and output dimension
    """
    
    def __init__(self, dim: int):
        super(FFLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 5 * dim),
            nn.ReLU(),
            nn.Linear(5 * dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x (torch.Tensor): Input of shape (B, N, dim)
        
        Returns:
            torch.Tensor: Output of shape (B, N, dim)
        """
        return self.net(x)


class AttentionBlock(nn.Module):
    """
    Transformer-style attention block with residual connections.
    
    Applies multi-head attention followed by feed-forward layer,
    each with layer normalization and residual connections.
    
    Args:
        head_size (int): Hidden dimension per attention head
        n_heads (int): Number of attention heads
    """
    
    def __init__(self, head_size: int, n_heads: int):
        super(AttentionBlock, self).__init__()
        embed_dim = n_heads * head_size
        
        self.mh = MultiHead(embed_dim, head_size, n_heads)
        self.ff = FFLayer(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention block.
        
        Args:
            x (torch.Tensor): Input of shape (B, G+T, n_heads*head_size)
            mask (torch.Tensor): Attention mask of shape (G+T, G+T)
        
        Returns:
            torch.Tensor: Output of shape (B, G+T, n_heads*head_size)
        """
        # Multi-head attention with residual
        x = x + self.mh(self.ln1(x), mask)
        
        # Feed-forward with residual
        x = x + self.ff(self.ln2(x))
        
        return x


class MultiBlockAttention(nn.Module):
    """
    Stacked attention blocks with final projection.
    
    Applies multiple attention blocks sequentially, then projects
    to the desired output dimension.
    
    Args:
        head_size (int): Hidden dimension per attention head
        n_heads (int): Number of attention heads
        out_dim (int): Output dimension
        n_att_blocks (int): Number of attention blocks to stack
    """
    
    def __init__(self, head_size: int, n_heads: int, out_dim: int, n_att_blocks: int):
        super(MultiBlockAttention, self).__init__()
        embed_dim = n_heads * head_size
        
        self.att_blocks = nn.ModuleList([
            AttentionBlock(head_size, n_heads)
            for _ in range(n_att_blocks)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacked attention blocks.
        
        Args:
            x (torch.Tensor): Input of shape (B, G+T, n_heads*head_size)
            mask (torch.Tensor): Attention mask of shape (G+T, G+T)
        
        Returns:
            torch.Tensor: Output of shape (B, G+T, out_dim)
        """
        # Apply attention blocks sequentially
        for block in self.att_blocks:
            x = block(x, mask)
        
        # Final layer norm and projection
        x = self.ln(x)
        x = self.head(x)
        
        return x


class AttGlobalAggregation(nn.Module):
    """
    Attention-based global aggregation of node embeddings.
    
    Aggregates all node embeddings into a single cell-level embedding
    using learned attention weights.
    
    Args:
        out_dim (int): Dimension of node embeddings to aggregate
    """
    
    def __init__(self, out_dim: int):
        super(AttGlobalAggregation, self).__init__()
        self.attention_layer = nn.Linear(out_dim, 1)
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Aggregate node embeddings using attention.
        
        Args:
            node_embeddings (torch.Tensor): Node features of shape (B, G+T, out_dim)
        
        Returns:
            torch.Tensor: Cell embedding of shape (B, out_dim)
        """
        # Compute attention scores for each node
        scores = self.attention_layer(node_embeddings)  # (B, G+T, 1)
        
        # Normalize with softmax across nodes
        attention_weights = F.softmax(scores, dim=1)  # (B, G+T, 1)
        
        # Weighted sum of node embeddings
        cell_embedding = (attention_weights * node_embeddings).sum(dim=1)  # (B, out_dim)
        
        return cell_embedding


class GraphAwareEncoder(nn.Module):
    """
    Graph-aware encoder for cell state embedding.
    
    Takes gene expression values and a regulatory network structure,
    creates node embeddings incorporating both information, applies
    graph attention layers, and aggregates to cell-level representations.
    
    Architecture:
        1. Embedding table for all nodes (genes + TFs)
        2. Project and add gene expression to gene node embeddings
        3. Apply multi-block graph attention
        4. Global attention aggregation to cell embedding
    
    Args:
        n_genes (int): Number of genes (G)
        n_tfs (int): Number of TF-only nodes not in gene list (T)
        head_size (int): Hidden dimension per attention head
        n_heads (int): Number of attention heads
        out_dim (int): Output cell embedding dimension (D)
        n_att_blocks (int): Number of attention block layers
    
    Example:
        >>> encoder = GraphAwareEncoder(
        ...     n_genes=1000, n_tfs=100,
        ...     head_size=16, n_heads=4,
        ...     out_dim=64, n_att_blocks=2
        ... )
        >>> X = torch.randn(32, 1000, 1)  # 32 cells, 1000 genes
        >>> mask = create_mask_from_dorothea(...)  # (1100, 1100)
        >>> cell_emb = encoder(X, mask)  # (32, 64)
    """
    
    def __init__(
        self,
        n_genes: int,
        n_tfs: int,
        head_size: int,
        n_heads: int,
        out_dim: int,
        n_att_blocks: int
    ):
        super(GraphAwareEncoder, self).__init__()
        
        self.n_genes = n_genes
        self.n_tfs = n_tfs
        self.n_nodes = n_genes + n_tfs
        
        embed_dim = n_heads * head_size
        
        # Embedding table for all nodes (genes + TFs)
        self.node_embeddings = nn.Embedding(self.n_nodes, embed_dim)
        
        # Project gene expression to embedding space
        self.expr_projection = nn.Linear(1, embed_dim)
        
        # Multi-block attention
        self.multi_block_att = MultiBlockAttention(
            head_size, n_heads, out_dim, n_att_blocks
        )
        
        # Global aggregation
        self.global_agg = AttGlobalAggregation(out_dim)
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from gene expression to cell embedding.
        
        Args:
            X (torch.Tensor): Gene expression values of shape (B, G, 1)
            mask (torch.Tensor): Graph structure mask of shape (G+T, G+T)
                                Float weights for edges, -inf for no edges
        
        Returns:
            torch.Tensor: Cell embeddings of shape (B, out_dim)
        """
        B = X.shape[0]
        
        # Get embedding table
        E = self.node_embeddings.weight  # (G+T, embed_dim)
        
        # Project gene expression and add to gene embeddings
        X_proj = self.expr_projection(X)  # (B, G, embed_dim)
        gene_emb = E[:self.n_genes] + X_proj  # (B, G, embed_dim) - broadcasting handles it
        
        # Get TF-only embeddings and expand for batch
        tf_emb = E[self.n_genes:].unsqueeze(0).expand(B, -1, -1)  # (B, T, embed_dim)
        
        # Concatenate gene and TF embeddings
        all_emb = torch.cat([gene_emb, tf_emb], dim=1)  # (B, G+T, embed_dim)
        
        # Apply attention blocks with graph structure
        node_features = self.multi_block_att(all_emb, mask)  # (B, G+T, out_dim)
        
        # Global aggregation to cell-level embedding
        cell_emb = self.global_agg(node_features)  # (B, out_dim)
        
        return cell_emb
