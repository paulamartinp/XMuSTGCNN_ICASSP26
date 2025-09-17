# GRU-GCNN Baseline

This folder contains the implementation of the **GRU-GCNN** baseline model used 
for comparison with XMuST-GCNN.

---

## Structure

```text
GRU_GCNN/
├─ experiments/             # Training and evaluation pipeline
│  ├─ hyperparameters/      # Selected hyperparameters
│  ├─ models/               # Saved model checkpoints (not uploaded to avoid large files)
│  ├─ results/              # Inference and evaluation results
│  └─ load_config.json      # Default experiment configuration
│
├─ optimize.py              # Hyperparameter search / optimization
├─ test.py                  # Evaluation script
├─ model.py                 # GRU-GCNN architecture definition
└─ utils_baseline.py        # Utility functions for training/inference
