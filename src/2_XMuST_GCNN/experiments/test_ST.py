import torch
import pandas as pd
import json
import numpy as np
import os
from typing import Dict, Any
import sys

sys.path.append('../')
import utils_ST as utils

def load_params_from_json(filepath: str) -> Dict[str, Any]:
    """Load parameters from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    # 1) Load experiment configuration
    params = load_params_from_json("load_config_ST.json")
    device = torch.device(params["device"])
    graph_method = params["way_to_build_graph"]
    folders = params["folders"]
    all_results = {}
    all_intrinsic_xai: Dict[str, Any] = {}
    for folder, seed in zip(params["folders"], params['seed']):
        print(f"\n=== folder {folder} ===")

        (X_train_temporal, X_val_temporal, X_test_temporal,
         X_train_static,   X_val_static,   X_test_static,
         y_train,          y_val,          y_test) = utils.load_data(
            device, folder,
            params["numberOfTimeStep"],
            params["SG"]
        )

        # 3) Load the static adjacency (one matrix)
        static_path = (
            f"../../../graph_estimation/static_data/"
            f"step2_graphRepresentation/{graph_method}/s{folder}/"
            f"{params['path_S_static']}"
        )
        S_static = pd.read_csv(static_path).values
        S_static = torch.tensor(S_static, dtype=torch.float32).to(device)



        temp_path = (f"../../../graph_estimation/temporal_data/step2_graphRepresentation/{graph_method}/s{folder}/SpaceTimeGraph_Xtr_minMax_per_patient_th_0.975.csv"

        )
        S_temporal = pd.read_csv(temp_path).values
        S_temporal = torch.tensor(S_temporal, dtype=torch.float32).to(device)

        # 5) Load best hyperparameters for this split
        hyperparams_path = f"./hyperparameters/{graph_method}/best_params_gnn_ST.json"
        best_hps = utils.loadBestHyperparameters(hyperparams_path)[folder]

        # 6) Initialize TestModel with your updated class signature
        tester = utils.TestModel(
            seed=seed,
            model_class=utils.MultimodalGNN,
            device=device,
            best_model_path=f"./models/model_ST_{folder}.pth"
        )

        # 7) Run the test routine
        results, intrinsic_xai = tester.test(
            hyperparams_path=hyperparams_path,
            split=folder,
            S_temporal_list=S_temporal,
            S_static=S_static,
            X_train_temporal=X_train_temporal,
            X_train_static=X_train_static,
            y_train=y_train,
            X_val_temporal=X_val_temporal,
            X_val_static=X_val_static,
            y_val=y_val,
            X_test_temporal=X_test_temporal,
            X_test_static=X_test_static,
            y_test=y_test,
            n_epochs=best_hps["train_params"]["n_epochs"],
            early_stopping_patience=best_hps["train_params"]["early_stopping_patience"]
        )


        all_results[folder] = results
        print(f"Split {folder} results:", results)

        # --- Convert and store this folder's XAI outputs ---
        serializable = {
            key: value.detach().cpu().numpy().tolist()
            for key, value in intrinsic_xai.items()
        }
        all_intrinsic_xai[folder] = serializable
        # --- Remove unwanted entries from intrinsic_xai ---
        intrinsic_xai.pop("probability", None)
        intrinsic_xai.pop("static_embedding", None)
        intrinsic_xai.pop("temporal_embedding", None)

    # 9) Write aggregated test metrics
    final_path = "./results/metrics_by_split_ST.json"
    with open(final_path, "w") as mf:
        json.dump(all_results, mf, indent=4)
    print(f"\nAll test metrics saved to {final_path}")

    metrics = [
    "test_acc", "roc_auc", "aucpr", "f1_score",
    "sensitivity", "specificity",
    "inference_time", "training_time"
]

    df = pd.DataFrame.from_dict(all_results, orient="index") 
    summary = df[metrics].agg(["mean", "std"])

    print("\n=== Summary of all splits ===")
    print(summary.round(4))  

    

def clean_up_gpu():
    """Empty CUDA cache and collect garbage."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

if __name__ == "__main__":
    main()
    clean_up_gpu()
