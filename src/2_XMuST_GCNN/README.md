# XMuST-GCNN

This folder contains the implementation of **XMuST-GCNN** (eXplainable Multimodal 
Spatio-Temporal Graph Convolutional Neural Network)

---

## Structure

```text
2_XMuST_GCNN/
├─ experiments/                  # Training and evaluation pipeline
│  ├─ hyperparameters/           # Hyperparameter selection
│  ├─ models/                    # Saved model checkpoints
│  ├─ results/                   # Inference and evaluation results
│  └─ load_config_ST.json        # Default experiment configuration
│
├─ optimization_ST.py            # Script for hyperparameter optimization
├─ optuna_search_space.json      # Search space definition for Optuna
├─ test_ST.py                    # Evaluation script
│
├─ explainability_outputs/       # Outputs of explainability experiments
├─ explainer.py                  # Extension of GNNExplainer to multimodal setting
├─ plot_explainer.py             # Utilities for plotting explanations
│
├─ model.py                      # XMuST-GCNN architecture definition
├─ utils_ST.py                   # Utility functions for training/inference
