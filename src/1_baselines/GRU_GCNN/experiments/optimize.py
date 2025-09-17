import torch
import pandas as pd
import json
import os
from typing import Dict, Any, List

# My libraries
import sys
sys.path.append('../')  
import utils_baseline as utils

def load_params_from_json(filepath: str) -> Dict[str, Any]:
    """Load parameters from JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)

def main():
    # Load configuration
    params = load_params_from_json("./load_config.json")
    
    device = torch.device(params["device"])
    folders = params["folders"]
    way_to_build_graph = params["way_to_build_graph"]
    n_trials = params["n_trials"]
    
    best_result_by_split = {}
    optimization_time_by_split = {}

    for folder in folders:
        print(f"\n=== Processing folder {folder} ===")
        print(f"FOLDER INSIDE FUNCTION {folder}")
        torch.cuda.empty_cache()

        # Load multimodal data
        (X_train_temporal, X_val_temporal, X_test_temporal,
         X_train_static, X_val_static, X_test_static,
         y_train, y_val, y_test) = utils.load_data(device, folder, params["numberOfTimeStep"], params["SG"])
        
        # Load adjacency matrices
        # Static graph (single matrix)
        S_static = pd.read_csv(
            f"../../../../graph_estimation/static_data/step2_graphRepresentation/{way_to_build_graph}/s{folder}/{params['path_S_static']}"
        )
        S_static = torch.tensor(S_static.values, dtype=torch.float32).to(device)

        # We do not use the ST graph here
        ST_graph = []

        # Initialize optimizer
        optimizer = utils.Optimizer(device=device)
        # Run optimization
        for s in range(1,11):
            seed = params["seed"][s-1]
        
            best_hyperparameters, optimization_time = optimizer.optimize_hyperparameters(
                S_temporal_list=ST_graph,  
                S_static=S_static,               
                X_train_temporal=X_train_temporal,  
                X_train_static=X_train_static,     
                X_val_temporal=X_val_temporal,  
                X_val_static=X_val_static,        
                y_train=y_train,
                y_val=y_val,
                params=params,
                seed=seed,
                n_trials=n_trials
            
            )
        
            best_result_by_split[folder] = best_hyperparameters
            optimization_time_by_split[folder] = optimization_time

    # Save results
    output_dir = f"./hyperparameters/{way_to_build_graph}"
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "baseline.json")
    utils.saveBestHyperparameters(best_result_by_split, output_filepath, params)

def clean_up_gpu():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  

if __name__ == "__main__":
    main()
    clean_up_gpu()
