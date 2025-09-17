# XMuST-GCNN: eXplainable Multimodal Spatio-Temporal Graph Convolutional Neural Network  

> **ICASSP 2026 Submission**  
> A novel framework for modeling **heterogeneous clinical data** with multimodal graph neural networks, balancing performance and interpretability.  

---

## 🌟 Highlights  

- **Multimodal fusion**: Combines sequential (time-varying) and static (time-invariant) patient data.  
- **Spatio-temporal graphs**: Model irregular multivariate time series via dynamic graph construction.  
- **Parallel static graph**: Encode demographic + invariant features.  
- **Gated attention mechanism**: Learns how to balance contributions from temporal and static modalities.  
- **Explainability**: Extension of **GNNExplainer** to multimodal settings, identifying feature–time interactions and static risk factors.  
- **Performance**: Achieved **AUCROC = 82.15 ± 1.33%** on ICU antimicrobial resistance (AMR) prediction, surpassing strong graph-based baselines by >4 points.  

---

## 📂 Repository Structure  

```text
├─ DATA/                 # Placeholder (confidential patient data not released)
├─ graph_estimation/     # Scripts + notebooks for graph construction
├─ src/                  # Source code
│  ├─ 1_baselines/       # Baseline models for comparison
│  ├─ 2_XMuST_GCNN/      # Core implementation of XMuST-GCNN + multimodal GNNExplainer
│  └─ results_inference/ # Inference + evaluation scripts
└─ README.md             # You are here 
