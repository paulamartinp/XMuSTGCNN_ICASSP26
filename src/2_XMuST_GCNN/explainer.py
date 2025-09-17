#!/usr/bin/env python3
"""
Usage:
    python explain_mdr_masks.py \
        --config path/to/config.json \
        --model_module model \
        --model_class MultimodalGNN \
        --weights path/to/weights.pt \
        --model_name experiment1 \
        --folder 1 \
        --output_dir results \
        --explainer_epochs 300 \
        --explainer_lr 0.01

Ensure that your parent directory (where models_gnn_rnn_*.py live) is on PYTHONPATH. By default, this script will add two levels up to the path, so if your models are located at ../../models_gnn_rnn_1.py, you can import them as:

    --model_module models_gnn_rnn_1
    --model_class MultimodalGNN

"""

import argparse
import json
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import importlib
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any



def load_data(device: torch.device, split: str, numberOfTimeStep:int, SG: str) -> tuple:
    """
    Load and preprocess static and temporal data for training, validation, and testing.

    Parameters
    ----------
    device : torch.device
        Device to load the data onto (CPU or GPU).
    split : str
        Data split name (e.g., 'train', 'val', 'test').
    SG : bool
        Whether to apply feature-wise temporal averaging and reshape to (n, F×T).

    Returns
    -------
    tuple
        Preprocessed tensors for temporal/static features and labels.
    """
    print('FOLDER INSIDE FUNCTION', split)
    
    # Load static data
    X_train_static = pd.read_csv(f"../../DATA/s{split}/X_train_static.csv")
    X_train_static = X_train_static.to_numpy()
    X_train_static = X_train_static.reshape(X_train_static.shape[0], 1, X_train_static.shape[1])
    
    X_val_static = pd.read_csv(f"../../DATA/s{split}/X_val_static.csv")
    X_val_static = X_val_static.to_numpy()
    X_val_static = X_val_static.reshape(X_val_static.shape[0], 1, X_val_static.shape[1])

    X_test_static = pd.read_csv(f"../../DATA/s{split}/X_test_static.csv")
    X_test_static = X_test_static.to_numpy()
    X_test_static = X_test_static.reshape(X_test_static.shape[0], 1, X_test_static.shape[1])


    # Load temporal data
    X_train_temporal = np.load(f"../../DATA/s{split}/X_train_tensor_minMax_per_patient.npy")
    X_val_temporal = np.load(f"../../DATA/s{split}/X_val_tensor_minMax_per_patient.npy")
    X_test_temporal = np.load(f"../../DATA/s{split}/X_test_tensor_minMax_per_patient.npy")

    # Load labels
    y_train = pd.read_csv(f"../../DATA/s{split}/y_train_tensor_minMax_per_patient.csv", index_col=0)
    y_val = pd.read_csv(f"../../DATA/s{split}/y_val_tensor_minMax_per_patient.csv", index_col=0)
    y_test = pd.read_csv(f"../../DATA/s{split}/y_test_tensor_minMax_per_patient.csv", index_col=0)

    y_train = pd.read_csv(f"../../DATA/s{split}/y_train_tensor_minMax_per_patient.csv")[['MR_stac']]
    y_train = y_train.iloc[0:y_train.shape[0]:numberOfTimeStep].reset_index(drop=True)
    y_train = torch.tensor(y_train['MR_stac'], dtype=torch.float32)


    y_val = pd.read_csv(f"../../DATA/s{split}/y_val_tensor_minMax_per_patient.csv")[['MR_stac']]
    y_val = y_val.iloc[0:y_val.shape[0]:numberOfTimeStep].reset_index(drop=True)
    y_val = torch.tensor(y_val['MR_stac'], dtype=torch.float32)

    y_test = pd.read_csv(f"../../DATA/s{split}/y_test_tensor_minMax_per_patient.csv")[['MR_stac']]
    y_test = y_test.iloc[0:y_test.shape[0]:numberOfTimeStep].reset_index(drop=True)
    y_test = torch.tensor(y_test['MR_stac'], dtype=torch.float32)

    # Replace missing values
    X_train_static[X_train_static == 666] = 0
    X_val_static[X_val_static == 666] = 0
    X_test_static[X_test_static == 666] = 0

    if SG == True:
        # Replace 666 with NaN for temporal
        X_train_temporal[X_train_temporal == 666] = 0
        X_val_temporal[X_val_temporal == 666] = 0
        X_test_temporal[X_test_temporal == 666] = 0

        # Vectorize each of the train/val/test sets
        n, dim1, dim2 = X_train_temporal.shape
        X_train_temporal = X_train_temporal.reshape((n, dim1 * dim2))

        n, dim1, dim2 = X_val_temporal.shape
        X_val_temporal = X_val_temporal.reshape((n, dim1 * dim2))

        n, dim1, dim2 = X_test_temporal.shape
        X_test_temporal = X_test_temporal.reshape((n, dim1 * dim2))

        X_train_temporal = torch.tensor(X_train_temporal, dtype=torch.float32)  # (P, F*T)
        X_val_temporal = torch.tensor(X_val_temporal, dtype=torch.float32)
        X_test_temporal = torch.tensor(X_test_temporal, dtype=torch.float32)

        X_train_temporal = X_train_temporal.unsqueeze(2)
        X_val_temporal = X_val_temporal.unsqueeze(2)
        X_test_temporal = X_test_temporal.unsqueeze(2)

    else:
        # Replace 666 with 0
        X_train_temporal[X_train_temporal == 666] = 0
        X_val_temporal[X_val_temporal == 666] = 0
        X_test_temporal[X_test_temporal == 666] = 0

        # Convert to tensors
        X_train_temporal = torch.tensor(X_train_temporal, dtype=torch.float32)
        X_val_temporal = torch.tensor(X_val_temporal, dtype=torch.float32)
        X_test_temporal = torch.tensor(X_test_temporal, dtype=torch.float32)

        # X_train_temporal = X_train_temporal.unsqueeze(2)
        # X_val_temporal = X_val_temporal.unsqueeze(2)
        # X_test_temporal = X_test_temporal.unsqueeze(2)

    # Convert static
    n, dim1, dim2 = X_train_static.shape
    X_train_static = X_train_static.reshape((n, dim2, dim1))

    n, dim1, dim2 = X_val_static.shape
    X_val_static = X_val_static.reshape((n, dim2, dim1))

    n, dim1, dim2 = X_test_static.shape
    X_test_static = X_test_static.reshape((n, dim2, dim1))

    X_train_static = torch.tensor(X_train_static, dtype=torch.float32)
    X_val_static = torch.tensor(X_val_static, dtype=torch.float32)
    X_test_static = torch.tensor(X_test_static, dtype=torch.float32)

    # Move to device
    if device.type == "cuda":
        return (
            X_train_temporal.to(device), X_val_temporal.to(device), X_test_temporal.to(device),
            X_train_static.to(device), X_val_static.to(device), X_test_static.to(device),
            y_train.to(device), y_val.to(device), y_test.to(device)
        )
    else:
        return (
            X_train_temporal, X_val_temporal, X_test_temporal,
            X_train_static, X_val_static, X_test_static,
            y_train, y_val, y_test
        )
    

def load_params_from_json(filepath: str) -> Dict[str, Any]:
    """Load parameters from JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and save mean temporal/static masks and visualizations for MDR and non-MDR classes"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to JSON config with data/model parameters"
    )
    parser.add_argument(
        "--model_module", type=str, required=True,
        help="Python module path where the model class is defined, e.g. 'models.my_model'"
    )
    parser.add_argument(
        "--model_class", type=str, required=True,
        help="Name of the model class in the module, e.g. 'MultimodalGNN'"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Optional path to pretrained model weights"
    )
    parser.add_argument(
        "--model_name", type=str, default='model',
        help="Identifier for this model (used for its output folder)"
    )
    parser.add_argument(
        "--folder", type=str, default='1',
        help="Data folder index (e.g., '1', '2', ...)")
    parser.add_argument(
        "--explainer_epochs", type=int, default=300,
        help="Epochs for explainer optimization"
    )
    parser.add_argument(
        "--explainer_lr", type=float, default=0.01,
        help="Learning rate for explainer optimizer"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default='cuda',
        help="Device to run on: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--output_dir", type=str, default='explainability_outputs',
        help="Base directory to save masks and visualizations"
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultimodalFeatureMaskExplainer(nn.Module):
    def __init__(self, model: nn.Module, temporal_dim: int, static_dim: int):
        super().__init__()
        self.model = model
        self.temporal_mask = nn.Parameter(torch.randn(temporal_dim))
        self.static_mask = nn.Parameter(torch.randn(static_dim, 1))
        self.loss_fn = nn.BCELoss()

    def forward(self, x_temporal, x_static: torch.Tensor) -> torch.Tensor:
        train_mode = self.model.training
        self.model.train()
        orig_req = {n: p.requires_grad for n, p in self.model.named_parameters()}
        for p in self.model.parameters(): 
            p.requires_grad_(False)
        try:
            # Static masking
            sm = torch.sigmoid(self.static_mask)
            x_stat_masked = x_static * sm
            if x_stat_masked.dim() == 2:
                x_stat_masked = x_stat_masked.unsqueeze(0)

            # Temporal masking
            if isinstance(x_temporal, list):
                tm = torch.sigmoid(self.temporal_mask).view(len(x_temporal), -1)
                x_temp_masked = [x * tm[t].view(1, -1, 1) for t, x in enumerate(x_temporal)]
            else:
                tm = torch.sigmoid(self.temporal_mask).view(1, -1, 1)
                x_temp_masked = x_temporal * tm



            out, _, _ = self.model(x_temp_masked, x_stat_masked)
            return out.squeeze(-1)
        finally:
            for n, p in self.model.named_parameters():
                p.requires_grad_(orig_req[n])
            self.model.train(train_mode)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                     entropy_weight=0.1, l1_weight=0.005) -> torch.Tensor:
        pred_loss = self.loss_fn(y_pred, y_true)
        t_sig = torch.sigmoid(self.temporal_mask)
        s_sig = torch.sigmoid(self.static_mask)
        ent_t = -(t_sig * torch.log(t_sig+1e-8) + (1-t_sig) * torch.log(1-t_sig+1e-8)).mean()
        ent_s = -(s_sig * torch.log(s_sig+1e-8) + (1-s_sig)* torch.log(1-s_sig+1e-8)).mean()
        l1 = t_sig.mean() + s_sig.mean()
        return pred_loss + entropy_weight*(ent_t + ent_s) + l1_weight*l1


def explain_instance(
    explainer: MultimodalFeatureMaskExplainer,
    x_temp_list: List[torch.Tensor],
    x_static: torch.Tensor,
    y_true: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int
) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = explainer(x_temp_list, x_static)
        loss = explainer.compute_loss(pred, y_true, entropy_weight=0.0, l1_weight=0.0)
        loss.backward()
        optimizer.step()
    return explainer.temporal_mask.sigmoid().cpu().detach().numpy(), explainer.static_mask.sigmoid().cpu().detach().numpy()


def batch_explain(
    model: nn.Module,
    X_temp: torch.Tensor,
    X_stat: torch.Tensor,
    y: torch.Tensor,
    params: Dict,
    cfg,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_idx = (y.cpu().numpy().flatten()==1).nonzero()[0].tolist()
    neg_idx = (y.cpu().numpy().flatten()==0).nonzero()[0].tolist()
    n = min(len(pos_idx), len(neg_idx))
    selected = {
        'pos': random.sample(pos_idx, n),
        'neg': random.sample(neg_idx, n)
    }
    out = {}
    print(params.keys())
    for label, idxs in selected.items():
        masks_t, masks_s = [], []
        for i in tqdm(idxs, desc=f"Explaining {label} samples"):
            if cfg['SG'] == True:
                x_input = X_temp[i].unsqueeze(0).to(device)  # tensor: [1, F, 1]
                temporal_dim = 1190
            else:
                x_input = [X_temp[i, t].unsqueeze(0).unsqueeze(-1).to(device) for t in range(params['numberOfTimeStep'])]
                temporal_dim = 1190
            
            x_s = X_stat[i].to(device)
            y_i = y[i].unsqueeze(0).float().to(device)
            explainer = MultimodalFeatureMaskExplainer(model,
                        temporal_dim=1190,
                        static_dim=X_stat.shape[1]).to(device)
            optimizer = torch.optim.Adam(explainer.parameters(), lr=params['explainer_lr'])
            tm, sm = explain_instance(explainer, x_input, x_s, y_i, optimizer, params['explainer_epochs'])
            masks_t.append(tm)
            masks_s.append(sm)
        out[label] = (np.stack(masks_t).mean(axis=0), np.stack(masks_s).mean(axis=0))
    return out['pos'][0], out['neg'][0], out['pos'][1], out['neg'][1]


def save_visualizations(
    temporal_mask: np.ndarray,
    static_mask: np.ndarray,
    out_dir: str,
    time_steps: int,
    features_per_step: int,
    static_feature_names: List[str] = None,
    prefix: str = ''
):
    os.makedirs(out_dir, exist_ok=True)
    # Temporal heatmap
    tm2 = temporal_mask.reshape(time_steps, features_per_step)
    plt.figure(figsize=(10,4))
    plt.imshow(tm2, aspect='auto', cmap='coolwarm')
    plt.colorbar(label='Importance')
    plt.title(f"{prefix} Temporal Mask")
    plt.xlabel('Feature')
    plt.ylabel('Time Step')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_temporal_mask.png"))
    plt.close()

    # Static bar chart
    plt.figure(figsize=(6,3))
    vals = static_mask.flatten()
    names = static_feature_names or [f'S{i}' for i in range(len(vals))]
    plt.bar(names, vals)
    plt.title(f"{prefix} Static Mask")
    plt.xticks(rotation=45)
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_static_mask.png"))
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    print('WORKING!')
    # Load params and data
    cfg = json.load(open(args.config))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    (X_tr_temp, X_val_temp, X_test_temp,
     X_tr_stat, X_val_stat, X_test_stat,
     y_tr, y_val, y_test) = load_data(device, args.folder, cfg['numberOfTimeStep'], cfg['SG'])

    # Dynamically import model
    mod = importlib.import_module(args.model_module)
    ModelClass = getattr(mod, args.model_class)

    RNN_GNN_VARIANTS = (
        'models_rnn_gnn_1',
        'models_rnn_gnn_2',
        'models_gnn_rnn_1',
    )


    if args.model_module in RNN_GNN_VARIANTS:
        # build one S_temporal per timestep
        S_temporal = []
        for nt in range(cfg["numberOfTimeStep"]):
            path = (
                f"../../graph_estimation/temporal_data/"
                f"step2_graphRepresentation/{cfg['way_to_build_graph']}/"
                f"s{args.folder}/adj_X_train_{nt}_{cfg['norm']}_th_0.975.csv"
            )
            df = pd.read_csv(path)
            print(f"Step {nt}: {df.shape}")
            S_temporal.append(torch.tensor(df.values, dtype=torch.float32, device=device))
    else:
        # single combined graph for the SpaceTime variant
        path_S_temporal = "SpaceTimeGraph_Xtr_minMax_per_patient_th_0.975.csv"
        S_temporal = pd.read_csv(f"../../graph_estimation/temporal_data/step2_graphRepresentation/{cfg['way_to_build_graph']}/s{args.folder}/{path_S_temporal}")
        print(f" {S_temporal.shape}")
        S_temporal = torch.tensor(S_temporal.values, dtype=torch.float32).to(device)
        

    S_static = pd.read_csv(
    f"../../graph_estimation/static_data/step2_graphRepresentation/{cfg['way_to_build_graph']}/s{args.folder}/{cfg['path_S_static']}"
    )
    S_static = torch.tensor(S_static.values, dtype=torch.float32).to(device)

    cfg[str(args.folder)]['temporal_gnn_params']['seed'] = cfg['seed'][int(args.folder)-1]
    cfg[str(args.folder)]['static_gnn_params']['seed'] = cfg['seed'][int(args.folder)-1]
    model: nn.Module = ModelClass(
        cfg[str(args.folder)]['temporal_gnn_params'], cfg[str(args.folder)]['static_gnn_params'], S_temporal, S_static
    ).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    explainer_params = {
        'numberOfTimeStep': cfg['numberOfTimeStep'],
        'explainer_lr': args.explainer_lr,
        'explainer_epochs': args.explainer_epochs
    }

    # Explain on test set
    print('FORMA DE TEST', X_test_temp.shape)
    mean_t_pos, mean_t_neg, mean_s_pos, mean_s_neg = batch_explain(
        model, X_test_temp, X_test_stat, y_test, explainer_params, cfg, device
    )

    # Prepare output folders
    model_dir = os.path.join(args.output_dir, args.model_name)
    npy_dir = os.path.join(model_dir, 'npy')
    viz_dir = os.path.join(model_dir, 'visualizations')
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # Save numpy arrays
    np.save(os.path.join(npy_dir, 'mean_temporal_pos.npy'), mean_t_pos)
    np.save(os.path.join(npy_dir, 'mean_temporal_neg.npy'), mean_t_neg)
    np.save(os.path.join(npy_dir, 'mean_static_pos.npy'), mean_s_pos)
    np.save(os.path.join(npy_dir, 'mean_static_neg.npy'), mean_s_neg)

    # Save visualizations
    static_names = cfg.get('static_feature_names', None)
    temporal_mask_length = mean_t_pos.shape[0]                    # 1190
    time_steps         = cfg['numberOfTimeStep']                 # 14
    features_per_step  = temporal_mask_length // time_steps      # 1190//14 = 85

    save_visualizations(
        mean_t_pos,
        mean_s_pos,
        viz_dir,
        time_steps,
        features_per_step,
        static_names,
        prefix='pos'
    )

    save_visualizations(mean_t_neg, mean_s_neg, viz_dir, time_steps, features_per_step, static_names, prefix='neg')

    print(f"Saved masks (.npy) to {npy_dir}")
    print(f"Saved visualizations (.png) to {viz_dir}")


if __name__ == '__main__':
    main()
