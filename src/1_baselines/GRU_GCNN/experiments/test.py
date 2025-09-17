import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
sys.path.append('../')  
import utils_baseline as utils

def load_params(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)

def load_graphs(folder: str, params: Dict[str, Any], device: torch.device):
    way = params["way_to_build_graph"]
    # static adjacency
    S_static = pd.read_csv(
        f"../../../../graph_estimation/static_data/step2_graphRepresentation/"
        f"{way}/s{folder}/{params['path_S_static']}"
    )
    S_static = torch.tensor(S_static.values, dtype=torch.float32).to(device)
    # temporal adj
    S_temporal_list = []
    return S_static, S_temporal_list



def run_variant(variant: str, params, device, best_results):
    use_transformer = (variant.endswith("transformer"))
    suffix   = "" if not use_transformer else "_transformer"
    hyp_path = f"./hyperparameters/{params['way_to_build_graph']}/" \
               f"baseline.json"
    model_dir= f"./models/{variant}"
    result_dir = f"./results/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    hyper_all = utils.loadBestHyperparameters(hyp_path)
    for folder, seed in zip(params["folders"], params["seed"]):
        print(f"\n=== {variant} | folder {folder} ===")
        # 1) data
        X_tr_t, X_val_t, X_te_t, X_tr_s, X_val_s, X_te_s, y_tr, y_val, y_te = \
            utils.load_data(device, folder, params["numberOfTimeStep"], params["SG"])
        
        # 2) graphs + temporal lists
        S_static, S_temporal = load_graphs(folder, params, device)

        # 3) hyperparams
        hparams = hyper_all[folder]

        # 4) test
        tester = utils.TestModel(
            seed=seed,
            model_class=utils.MultimodalGNN,
            device=device,
            best_model_path=f"{model_dir}/model{folder}.pth"
        )
        res, temp_emb = tester.test(
            hyperparams_path=hyp_path,
            split=folder,
            S_temporal_list=S_temporal, # empty. here we are leveraging simply a GRU
            S_static=S_static,
            X_train_temporal=X_tr_t,
            X_train_static=X_tr_s,
            y_train=y_tr,
            X_val_temporal=X_val_t,
            X_val_static=X_val_s,
            y_val=y_val,
            X_test_temporal=X_te_t,
            X_test_static=X_te_s,
            y_test=y_te,
            n_epochs=hparams["train_params"]["n_epochs"],
            early_stopping_patience=hparams["train_params"]["early_stopping_patience"]
        )

        best_results[f"{folder}"] = res

    # save overall
    with open(f"{result_dir}/results.json", "w") as rf:
        json.dump(best_results, rf, indent=4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant",
        choices=["gnn_rnn_1"],
        required=True,
        help="Which model variant to run")
    p.add_argument("--config", default="load_config.json")
    args = p.parse_args()

    params = load_params(args.config)
    device = torch.device(params["device"])
    best_results = {}
    run_variant(args.variant, params, device, best_results)

    # summary
    print("\n=== Final Results ===")
    for key,res in best_results.items():
        print(f"{key}: ROC-AUC {res['roc_auc']:.4f}, Sens {res['sensitivity']:.4f}, Spec {res['specificity']:.4f}")

    # averages
    metrics = [
            "test_acc","roc_auc","aucpr","f1_score",
            "sensitivity","specificity",
            "inference_time","training_time","mean_weight"
        ]

    # collect into arrays
    arrs = {m: [] for m in metrics}
    for res in best_results.values():
        for m in metrics:
            arrs[m].append(res.get(m, np.nan))

    # print mean ± std
    print("\n=== Averages Across Folders ===")
    for m in metrics:
        vals = np.array(arrs[m], dtype=float)
        mean = np.nanmean(vals)
        std  = np.nanstd(vals)
        print(f"{m}: {mean:.4f} ± {std:.4f}")
if __name__=="__main__":
    main()
