import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_encoders.baseline_encoders import MLPEncoder
from baseline_encoders.decoder import Decoder, ProbLayer
from graph_aware_encoders.att_graph_aware_encoder import GraphAwareEncoder

class GraphModel(nn.Module):
    def __init__(self, architect_params: dict[str, any]):
        super().__init__()
        self.encoder = GraphAwareEncoder(
            n_genes=architect_params['n_genes'],
            n_tfs=architect_params['n_tfs'],
            n_heads=architect_params['n_heads'],
            head_size=architect_params['head_size'],
            n_att_blocks=architect_params['n_att_blocks'],
        )
        emb_dim = int(architect_params['head_size'] * architect_params['n_heads'])
        self.decoder = Decoder(
            in_dim=emb_dim,
            hidden_dim=5*emb_dim,
            out_dim=architect_params['x_dim']
        )
        self.probs = ProbLayer(emb_dim)

    def forward(self, x, mask):
        x_emb = self.encoder(x, mask)
        x_hat = self.decoder(x_emb)
        p = self.probs(x_emb)
        return x_emb, x_hat, p


class BaselineModel(nn.Module):
    def __init__(self, architect_params: dict[str, any]):
        super().__init__()
        self.encoder = MLPEncoder(
            in_dim=architect_params['x_dim'],
            hidden_dim=architect_params['hidden_dim'],
            n_layers= architect_params['n_layers'],
            emb_dim=architect_params['emb_dim'],
        )
        emb_dim = architect_params['emb_dim']

        self.decoder = Decoder(
            in_dim=emb_dim,
            hidden_dim=5*emb_dim,
            out_dim=architect_params['x_dim']
        )
        self.probs = ProbLayer(emb_dim)

    def forward(self, x):
        x_emb = self.encoder(x)
        x_hat = self.decoder(x_emb)
        p = self.probs(x_emb)
        return x_emb, x_hat, p