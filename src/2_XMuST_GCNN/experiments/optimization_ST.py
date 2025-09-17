import torch
import pandas as pd
import json
import os
from typing import Dict, Any

import sys
sys.path.append('../')  
import utils_ST as utils

def load_params_from_json(filepath: str) -> Dict[str, Any]:
    """Load configuration parameters from a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)


def main():
    # 1) Load configuration parameters
    params = load_params_from_json("./load_config_ST.json")
    
    device = torch.device(params["device"])
    splits = params["folders"]
    graph_method = params["way_to_build_graph"]
    
    best_results: Dict[str, Any] = {}
    runtime_by_split: Dict[str, float] = {}

    for split in splits:
        print(f"\n=== Processing split {split} ===")
        torch.cuda.empty_cache()

        # 2) Load multimodal features and labels
        (X_train_temporal, X_val_temporal, X_test_temporal,
         X_train_static,   X_val_static,   X_test_static,
         y_train,          y_val,          y_test) = utils.load_data(
            device, split,
            params["numberOfTimeStep"],
            params["SG"]
        )

        # 3) Load static adjacency matrix from CSV
        static_path = (
            f"../../../graph_estimation/static_data/step2_graphRepresentation/"
            f"{graph_method}/s{split}/{params['path_S_static']}"
        )
        S_static_df = pd.read_csv(static_path)
        S_static = torch.tensor(S_static_df.values, dtype=torch.float32).to(device)

        # 4) Load ST matrix
        path_S_temporal = f"../../../graph_estimation/temporal_data/step2_graphRepresentation/{graph_method}/s{split}/SpaceTimeGraph_Xtr_minMax_per_patient_th_0.975.csv"
        S_temp_df = pd.read_csv(path_S_temporal)
        S_temporal = torch.tensor(S_temp_df.values, dtype=torch.float32).to(device)

        # 5) Debug shapes to ensure correct loading
        print("Shapes:",
              f"Static adjacency={S_static.shape},",
              f"Temporal adjacency={S_temporal.shape},",
              f"X_train_temporal={X_train_temporal.shape},",
              f"X_train_static={X_train_static.shape},",
              f"y_train={y_train.shape}"
        )

        # 6) Initialize the optimizer and launch hyperparameter tuning
        for s in range(1,11):
            seed = params["seed"][s-1]
            optimizer = utils.Optimizer(device=device)
            best_params, runtime = optimizer.optimize_hyperparameters(
                S_temporal=S_temporal,
                S_static=S_static,
                X_train_temporal=X_train_temporal,
                X_train_static=X_train_static,
                X_val_temporal=X_val_temporal,
                X_val_static=X_val_static,
                y_train=y_train,
                y_val=y_val,
                params=params,
                seed=seed
            )

            best_results[split] = best_params
            runtime_by_split[split] = runtime

    # 7) Save best hyperparameters and runtimes
    output_dir = f"./hyperparameters/{graph_method}"
    os.makedirs(output_dir, exist_ok=True)

    utils.saveBestHyperparameters(
        best_results,
        os.path.join(output_dir, "best_params_gnn_ST.json"),
        params
    )
    # with open(os.path.join(output_dir, "runtime_by_split_gnn_ST.json"), "w") as f:
    #     json.dump(runtime_by_split, f, indent=4)

    print("Optimization complete. Results saved.")


def clean_up_gpu():
    """Free GPU memory and run garbage collection."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
    clean_up_gpu()
