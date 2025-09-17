import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self,
                 gnn_params_temporal: Dict,
                 gnn_params_static: Dict,

                 S_temporal: torch.Tensor,
                 S_static: torch.Tensor,
                 use_transformer: bool = False,
                 debug: bool = False):
        super().__init__()
        self.debug = debug
        self.use_transformer = use_transformer

        # Single temporal GNN
        self.temporal_gnn = StandardGCNN(gnn_params_temporal, S_temporal
        )

        # Static GNN
        self.static_gnn = StandardGCNN(gnn_params_static,S_static
        )

        combined_dim = gnn_params_temporal['embedding_dim'] + gnn_params_static['embedding_dim']
        self.attention = AttentionModule(
            static_dim=gnn_params_static['embedding_dim'],
            temporal_dim=gnn_params_temporal['embedding_dim'],
            fused_dim=combined_dim
        )

        self.fc = nn.Linear(combined_dim, 1)

    def _print_debug(self, *args):
        if self.debug:
            print(*args)

    def forward(self, x_temporal: torch.Tensor, x_static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self._print_debug("static input shape:", x_static.shape)
        self._print_debug("Temporal input shape:", x_temporal.mean(dim=1).unsqueeze(-1).shape)
        x_temporal = x_temporal.mean(dim=1).unsqueeze(-1)
        temporal_emb = self.temporal_gnn(x_temporal).squeeze(-1)


        # Static embedding from graph
        static_embedding = self.static_gnn(x_static).squeeze(-1)
        self._print_debug("Static embedding shape:", static_embedding.shape)

        # Fuse and predict
        fused, alpha = self.attention(static_embedding, temporal_emb)
        output = torch.sigmoid(self.fc(fused))
        return output, alpha, static_embedding


class MultimodalGNN_ST(nn.Module):
    def __init__(self,
                 gnn_params_temporal: Dict,
                 gnn_params_static: Dict,

                 S_temporal: torch.Tensor,
                 S_static: torch.Tensor,
                 use_transformer: bool = False,
                 debug: bool = False):
        super().__init__()
        self.debug = debug
        self.use_transformer = use_transformer

        # Single temporal GNN
        self.temporal_gnn = StandardGCNN(gnn_params_temporal, S_temporal
        )

        # Static GNN
        self.static_gnn = StandardGCNN(gnn_params_static,S_static
        )

        combined_dim = gnn_params_temporal['embedding_dim'] + gnn_params_static['embedding_dim']
        self.attention = AttentionModule(
            static_dim=gnn_params_static['embedding_dim'],
            temporal_dim=gnn_params_temporal['embedding_dim'],
            fused_dim=combined_dim
        )

        self.fc = nn.Linear(combined_dim, 1)

    def _print_debug(self, *args):
        if self.debug:
            print(*args)

    def forward(self, x_temporal: torch.Tensor, x_static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self._print_debug("Temporal input shape:", x_temporal.shape)
        temporal_emb = self.temporal_gnn(x_temporal).squeeze(-1)


        # Static embedding from graph
        static_embedding = self.static_gnn(x_static).squeeze(-1)
        self._print_debug("Static embedding shape:", static_embedding.shape)

        fused, alpha = self.attention(static_embedding, temporal_emb)
        importance_pre_fc = fused.clone()    

        logit = self.fc(fused)                                # (B,1)
        prob  = torch.sigmoid(logit)

        # 5) Pesos de la primera capa FC
        w_fc1 = self.fc.weight.data.clone()                   # (1, fused_dim)

        # 6) Salida
        return {
            'probability':       prob,       # P(MDR)
            'logits':            logit,      # pre-sigmoid
            'attention_alpha':   alpha,      # cuánto pesa estático vs temporal
            'static_embedding':  static_embedding, # embedding GNN estático
            'temporal_embedding':temporal_emb, # embedding GNN temporal
            'fc1_weights':       w_fc1,      # pesos de la capa final
            'importance_pre_fc': importance_pre_fc
        }


