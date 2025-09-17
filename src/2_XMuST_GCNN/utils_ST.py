import torch
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, average_precision_score  
import optuna    
import torch
import json
from model import MultimodalGNN_ST as MultimodalGNN
import torch.nn as nn
import time
from optuna.samplers import TPESampler
import torch.nn.functional as F
###############################################################
# Functions to save and load best hyperparameters and results #
###############################################################

def saveBestHyperparameters(best_hyperparameters, filename, original_params):
    import copy
    saved_params = copy.deepcopy(original_params)

    for split in best_hyperparameters:
        # Temporal GNN
        temporal_params = {
            "dropout": best_hyperparameters[split]["temporal_dropout"],
            "hid_dim": best_hyperparameters[split]["temporal_hid_dim"],
            "layers": best_hyperparameters[split]["temporal_layers"],
            "in_dim": original_params["temporal_gnn_params"]["in_dim"],
            "out_dim": original_params["temporal_gnn_params"]["out_dim"],
            "embedding_dim": original_params["temporal_gnn_params"]["embedding_dim"]
        }

        # Static GNN
        static_params = {
            "dropout": best_hyperparameters[split]["static_dropout"],
            "hid_dim": best_hyperparameters[split]["static_hid_dim"],
            "layers": best_hyperparameters[split]["static_layers"],
            "in_dim": original_params["static_gnn_params"]["in_dim"],
            "out_dim": original_params["static_gnn_params"]["out_dim"],
            "embedding_dim": original_params["static_gnn_params"]["embedding_dim"]
        }

        # Training params
        train_params = {
            "learning_rate": best_hyperparameters[split]["learning_rate"],
            "decay": best_hyperparameters[split]["decay"],
            "n_epochs": original_params["train_params"]["n_epochs"],
            "early_stopping_patience": original_params["train_params"]["early_stopping_patience"]
        }

        saved_params[split] = {
            "temporal_gnn_params": temporal_params,
            "static_gnn_params": static_params,
            "train_params": train_params
        }

    with open(filename, 'w') as file:
        json.dump(saved_params, file, indent=4)
        
def loadBestHyperparameters(filename):
    with open(filename, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters
#########################
# Function to load data #
#########################

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
    X_train_static = pd.read_csv(f"../../../DATA/s{split}/X_train_static.csv")
    X_train_static = X_train_static.to_numpy()
    X_train_static = X_train_static.reshape(X_train_static.shape[0], 1, X_train_static.shape[1])
    
    X_val_static = pd.read_csv(f"../../../DATA/s{split}/X_val_static.csv")
    X_val_static = X_val_static.to_numpy()
    X_val_static = X_val_static.reshape(X_val_static.shape[0], 1, X_val_static.shape[1])

    X_test_static = pd.read_csv(f"../../../DATA/s{split}/X_test_static.csv")
    X_test_static = X_test_static.to_numpy()
    X_test_static = X_test_static.reshape(X_test_static.shape[0], 1, X_test_static.shape[1])


    # Load temporal data
    X_train_temporal = np.load(f"../../../DATA/s{split}/X_train_tensor_minMax_per_patient.npy")
    X_val_temporal = np.load(f"../../../DATA/s{split}/X_val_tensor_minMax_per_patient.npy")
    X_test_temporal = np.load(f"../../../DATA/s{split}/X_test_tensor_minMax_per_patient.npy")

    # Load labels
    y_train = pd.read_csv(f"../../../DATA/s{split}/y_train_tensor_minMax_per_patient.csv", index_col=0)
    y_val = pd.read_csv(f"../../../DATA/s{split}/y_val_tensor_minMax_per_patient.csv", index_col=0)
    y_test = pd.read_csv(f"../../../DATA/s{split}/y_test_tensor_minMax_per_patient.csv", index_col=0)

    y_train = pd.read_csv(f"../../../DATA/s{split}/y_train_tensor_minMax_per_patient.csv")[['MR_stac']]
    y_train = y_train.iloc[0:y_train.shape[0]:numberOfTimeStep].reset_index(drop=True)
    y_train = torch.tensor(y_train['MR_stac'], dtype=torch.float32)


    y_val = pd.read_csv(f"../../../DATA/s{split}/y_val_tensor_minMax_per_patient.csv")[['MR_stac']]
    y_val = y_val.iloc[0:y_val.shape[0]:numberOfTimeStep].reset_index(drop=True)
    y_val = torch.tensor(y_val['MR_stac'], dtype=torch.float32)

    y_test = pd.read_csv(f"../../../DATA/s{split}/y_test_tensor_minMax_per_patient.csv")[['MR_stac']]
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


    

class Optimizer:
    """
    Entrenador y optimizador de hiperparámetros para MultimodalGNN_ST sin RNN.
    """
    def __init__(self, device, loss_function=nn.BCELoss(), storage=None, study_name=None):
        self.device = device
        self.loss_function = loss_function
        self.best_model_path = "best_multimodal_model.pth"
        self.storage = storage
        self.study_name = study_name

    def train(self, model, X_temporal, X_static, y, optimizer):
        model.train()
        optimizer.zero_grad()

        dict_output = model(X_temporal, X_static)
        output = dict_output['probability']
        bce_loss = F.binary_cross_entropy(output.squeeze(), y)

        total_loss = bce_loss

        total_loss.backward()
        optimizer.step()

        return total_loss

    def objective(self, trial, S_temporal, S_static,
                 X_train_temporal, X_train_static,
                 X_val_temporal, X_val_static,
                 y_train, y_val, params, seed):
        # Temporal GNN params
        temporal_gnn_params = {
            'dropout': trial.suggest_float('temporal_dropout',
                                          low=params['temporal_gnn_params']['dropout'][0],
                                          high=params['temporal_gnn_params']['dropout'][1]),
            'hid_dim': trial.suggest_int('temporal_hid_dim',
                                        low=params['temporal_gnn_params']['hid_dim'][0],
                                        high=params['temporal_gnn_params']['hid_dim'][1]),
            'layers': trial.suggest_int('temporal_layers',
                                        low=params['temporal_gnn_params']['layers'][0],
                                        high=params['temporal_gnn_params']['layers'][1]),
            'in_dim': params['temporal_gnn_params']['in_dim'],
            'out_dim': params['temporal_gnn_params']['out_dim'],
            'seed': seed,
            'embedding_dim': params['temporal_gnn_params']['embedding_dim']
        }

        # Static GNN params
        static_gnn_params = {
            'dropout': trial.suggest_float('static_dropout',
                                          low=params['static_gnn_params']['dropout'][0],
                                          high=params['static_gnn_params']['dropout'][1]),
            'hid_dim': trial.suggest_int('static_hid_dim',
                                        low=params['static_gnn_params']['hid_dim'][0],
                                        high=params['static_gnn_params']['hid_dim'][1]),
            'layers': trial.suggest_int('static_layers',
                                        low=params['static_gnn_params']['layers'][0],
                                        high=params['static_gnn_params']['layers'][1]),
            'in_dim': params['static_gnn_params']['in_dim'],
            'out_dim': params['static_gnn_params']['out_dim'],
            'seed': seed,
            'embedding_dim': params['static_gnn_params']['embedding_dim']
        }

        # Training params
        lr = trial.suggest_float('learning_rate',
                                 low=params['train_params']['learning_rate'][0],
                                 high=params['train_params']['learning_rate'][1],
                                 log=True)
        decay = trial.suggest_float('decay',
                                   low=params['train_params']['decay'][0],
                                   high=params['train_params']['decay'][1],
                                   log=True)

        # Model initialization
        model = MultimodalGNN(
            gnn_params_temporal=temporal_gnn_params,
            gnn_params_static=static_gnn_params,
            S_temporal=S_temporal,
            S_static=S_static,
            debug=False
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        # Training loop con early stopping
        best_roc_auc = -1
        early_stop_counter = 0
        for epoch in range(params['train_params']['n_epochs']):
            self.train(model, X_train_temporal, X_train_static, y_train, optimizer)
            model.eval()
            with torch.no_grad():
                dict_output = model(X_val_temporal, X_val_static)
                val_output = dict_output['probability'].squeeze()
                roc_auc = roc_auc_score(y_val.cpu().numpy(), val_output.cpu().numpy())
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    early_stop_counter = 0
                    torch.save(model.state_dict(), self.best_model_path)
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= params['train_params']['early_stopping_patience']:
                        break
        return best_roc_auc

    def optimize_hyperparameters(self, S_temporal, S_static,
                                X_train_temporal, X_train_static,
                                X_val_temporal, X_val_static,
                                y_train, y_val, params, seed):
        
        sampler = TPESampler(seed=seed, multivariate=True, group=True)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True
        )
        start = time.time()
        study.optimize(
            lambda trial: self.objective(
                trial, S_temporal, S_static,
                X_train_temporal, X_train_static,
                X_val_temporal, X_val_static,
                y_train, y_val, params, seed
            ),
            n_trials=params['n_trials']
        )
        return study.best_params, time.time() - start







class TestModel:
    def __init__(self, seed, model_class, device,
                 loss_function=nn.BCELoss(), best_model_path=None):
        self.seed = seed
        self.model_class = model_class
        self.device = device
        self.loss_function = loss_function
        self.best_model_path = best_model_path or f"./model_{seed}.pth"

    def load_hyperparameters(self, hyperparams_path, split):
        with open(hyperparams_path, 'r') as file:
            all_hyperparams = json.load(file)
        return all_hyperparams[split]

    def train_model(self, hyperparams, S_temporal, S_static,
                    X_train_temporal, X_train_static, y_train,
                    X_val_temporal, X_val_static, y_val,
                    n_epochs, early_stopping_patience):
        
        
        hyperparams["temporal_gnn_params"]['seed'] = self.seed
        hyperparams["static_gnn_params"]['seed'] = self.seed
        model = self.model_class(
            gnn_params_temporal=hyperparams["temporal_gnn_params"],
            gnn_params_static=hyperparams["static_gnn_params"],
            S_temporal=S_temporal,
            S_static=S_static
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparams["train_params"]["learning_rate"],
            weight_decay=hyperparams["train_params"]["decay"]
        )
        best_roc_auc = -1
        stop = 0
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()

            dict_output = model(X_train_temporal, X_train_static)
            output = dict_output['probability']
            loss = self.loss_function(output.squeeze(), y_train)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                dict_output = model(X_val_temporal, X_val_static)
                val_output = dict_output['probability'].squeeze()
                roc_auc = roc_auc_score(y_val.cpu().numpy(), val_output.squeeze().cpu().numpy())
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    stop = 0
                    torch.save(model.state_dict(), self.best_model_path)
                else:
                    stop += 1
                    if stop >= early_stopping_patience:
                        break
        model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        return model, None  

    def evaluate(self, model, X_test_temporal, X_test_static, y_test):
        model.eval()
        with torch.no_grad():
            start = time.time()
            dict_output = model(X_test_temporal, X_test_static)
            pred_probs = dict_output['probability'].squeeze()
            pred = torch.round(pred_probs).view(-1)
            tn, fp, fn, tp = confusion_matrix(y_test.cpu(), pred.cpu()).ravel()
        return {
            "test_acc": (pred == y_test).float().mean().item(),
            "roc_auc": roc_auc_score(y_test.cpu(), pred_probs.cpu()),
            "aucpr": average_precision_score(y_test.cpu(), pred_probs.cpu()),
            "f1_score": f1_score(y_test.cpu(), pred.cpu()),
            "sensitivity": tp/(tp+fn) if tp+fn>0 else 0,
            "specificity": tn/(tn+fp) if tn+fp>0 else 0,
            "inference_time": time.time()-start,
            "pred_probs": pred_probs.cpu().numpy().tolist(),
            "pred_labels": pred.cpu().numpy().tolist(),
        }, dict_output



    def test(self, hyperparams_path, split, S_temporal_list, S_static,
            X_train_temporal, X_train_static, y_train,
            X_val_temporal, X_val_static, y_val,
            X_test_temporal, X_test_static, y_test,
            n_epochs, early_stopping_patience):
        hyperparams = self.load_hyperparameters(hyperparams_path, split)

        model, training_time = self.train_model(
            hyperparams, S_temporal_list, S_static,
            X_train_temporal, X_train_static, y_train,
            X_val_temporal, X_val_static, y_val,
            n_epochs, early_stopping_patience
        )

        results, dict_output = self.evaluate(model, X_test_temporal, X_test_static, y_test)
        results["training_time"] = training_time
        return results, dict_output
