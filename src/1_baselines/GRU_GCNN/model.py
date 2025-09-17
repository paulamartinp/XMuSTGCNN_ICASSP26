import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple, Dict

class StandardGCNNLayer(nn.Module):
    """
    Standard Graph Convolutional Neural Network (GCN) layer.

    Parameters
    ----------
    S : torch.Tensor
        Normalized adjacency matrix of shape (N, N).
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    seed : int
        Random seed for weight initialization.
    """
    def __init__(self, S: torch.Tensor, in_dim: int, out_dim: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.S = S.clone()
        
        self.S += torch.eye(self.S.shape[0], device=self.S.device)
        self.d = self.S.sum(1)
        eps = 1e-6
        self.D_inv = torch.diag(1 / torch.sqrt(self.d + eps))
        self.S = self.D_inv @ self.S @ self.D_inv
        self.S = nn.Parameter(self.S, requires_grad=False)

        self.W = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        nn.init.kaiming_uniform_(self.W.data, nonlinearity='relu')
        
        self.b = nn.Parameter(torch.empty(self.out_dim))
        std = 1 / (self.in_dim * self.out_dim)
        nn.init.uniform_(self.b.data, -std, std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GCN layer.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape (N, in_dim).

        Returns
        -------
        torch.Tensor
            Output features of shape (N, out_dim).
        """
        return self.S @ x @ self.W + self.b[None, :]

class StandardGCNN(nn.Module):
    """
    Multi-layer Graph Convolutional Neural Network (GCN).

    Parameters
    ----------
    gnn_params : dict
        Dictionary containing hyperparameters for the GCN.
    S : torch.Tensor
        Normalized adjacency matrix.
    """
    def __init__(self, gnn_params: Dict, S: torch.Tensor):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.nonlin =  nn.ReLU()
        self.n_layers = gnn_params['layers']
        self.dropout = gnn_params['dropout']
        in_dim = gnn_params['in_dim']
        hid_dim = gnn_params['hid_dim']
        out_dim = gnn_params['out_dim']
        seed = gnn_params['seed']

        if self.n_layers > 1:
            self.convs.append(StandardGCNNLayer(S, in_dim, hid_dim, seed))
            for _ in range(self.n_layers - 2):
                in_dim = hid_dim
                self.convs.append(StandardGCNNLayer(S, in_dim, hid_dim, seed))
            in_dim = hid_dim
            self.convs.append(StandardGCNNLayer(S, in_dim, out_dim, seed))
        else:
            self.convs.append(StandardGCNNLayer(S, in_dim, out_dim, seed))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-layer GCN.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape (N, in_dim).

        Returns
        -------
        torch.Tensor
            Output features of shape (N, out_dim).
        """
        for i in range(self.n_layers - 1):
            x = self.nonlin(self.convs[i](x))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class AttentionModule(nn.Module):
    def __init__(self, static_dim, temporal_dim, fused_dim):
        super().__init__()
        self.static_proj = nn.Linear(static_dim, fused_dim)
        self.temporal_proj = nn.Linear(temporal_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, static, temporal):
        static = self.static_proj(static)
        temporal = self.temporal_proj(temporal)

        concat = torch.cat([static, temporal], dim=-1)
        alpha = self.gate(concat)  # (batch, 1)

        fused = alpha * static + (1 - alpha) * temporal
        return fused, alpha


class MultimodalGNN(nn.Module):
    def __init__(self, gnn_params_temporal: Dict, gnn_params_static: Dict,
                 rnn_params: Dict, S_temporal: List[torch.Tensor],
                 S_static: torch.Tensor, use_transformer: bool = False, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.use_transformer = use_transformer

        self.temporal_gnns = nn.ModuleList([
            StandardGCNN(gnn_params_temporal, S) for S in S_temporal
        ])

        temporal_gnn_out_dim = gnn_params_temporal['embedding_dim']

        self.rnn = nn.GRU(
            input_size=temporal_gnn_out_dim,
            hidden_size=temporal_gnn_out_dim,
            num_layers=rnn_params['num_layers'],
            bidirectional=rnn_params['bidirectional'],
            dropout=rnn_params['dropout'] if rnn_params['num_layers'] > 1 else 0,
            batch_first=True
        )
        
        self.temporal_out_dim = rnn_params['hidden_dim'] * (2 if rnn_params['bidirectional'] else 1)
        self.static_gnn = StandardGCNN(gnn_params_static, S_static)
        static_out_dim = gnn_params_static['embedding_dim']

        self.attention = AttentionModule(
            static_dim=static_out_dim,
            temporal_dim=1190,
            fused_dim=1190 + static_out_dim
        )

        self.fc = nn.Linear(1190 + static_out_dim, 1)
        self.attention_module = self.attention


    def _print_debug(self, *args):
        if self.debug:
            print(*args)

    def forward(self, x_temporal_list: List[torch.Tensor], x_static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x_seq = x_temporal_list  # (batch, time, emb)

        if not self.use_transformer:
            self._print_debug(f"RNN input shape: {x_seq.shape}")
            rnn_out, _ = self.rnn(x_seq)
            B, T, F = rnn_out.shape
            temporal_embedding = rnn_out.reshape(B, T*F)

        else:
            self._print_debug(f"Mamba input shape: {x_seq.shape}")
            mamba_out = self.temporal_encoder(x_seq)
            temporal_embedding = mamba_out.reshape(B, T*F)


        static_embedding = self.static_gnn(x_static).squeeze(-1)

        return temporal_embedding, static_embedding
    
    def classify(self, z_static, z_temporal):
        fused, alpha = self.attention(z_static, z_temporal) 
        return torch.sigmoid(self.fc(fused))

