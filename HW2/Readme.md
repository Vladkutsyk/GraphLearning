# Graph Machine Learning: Node Classification Comparison

This repository contains the implementation and evaluation of three different models for node classification. The goal was to compare how graph structure and node features contribute to the model's performance.

## 📌 Project Overview

We compared three distinct approaches:
1.  **DeepWalk (Graph Only):** Learned embeddings based strictly on graph topology using random walks and Word2Vec.
2.  **MLP (Features Only):** A standard 2-layer neural network using only node features for classification.
3.  **Vanilla GNN (Graph + Features):** A custom 2-layer Graph Neural Network implementation aggregating local neighborhoods.

## 📊 Results & Evaluation

The models were tested on the same dataset. Combining topology and features in the GNN yielded the best results.

| Approach | Information Used | Test Accuracy |
| :--- | :--- | :---: |
| 1. DeepWalk | Graph Topology Only | 71.60% |
| 2. MLP | Node Features Only | 53.40% |
| **3. Vanilla GNN** | **Graph + Features** | **75.10%** |

## 🌌 Visualizations (t-SNE)

<img width="1489" height="495" alt="image" src="https://github.com/user-attachments/assets/e0988b14-f46f-45da-9c6e-596f53ca70ff" />

We visualized the node embeddings using t-SNE to compare how well each model clusters different classes:

* **DeepWalk:** Clusters nodes based on graph proximity. Shows reasonable separation but lacks feature awareness.
* **MLP:** Embeddings are scattered. Without structural context, features alone are insufficient for clear class separation.
* **Vanilla GNN:** Shows the most distinct and clean color clusters. By performing message passing, the GNN successfully grouped nodes of the same class.

## 🧠 Key Technical Implementations

* **Custom Adjacency Builder:** Efficiently creates a dense adjacency matrix with self-loops from a PyG `edge_index`.
* **Vanilla GNN Layer:** Implemented the message passing step $H^{(k+1)} = \hat{A} \cdot (H^{(k)} W^{(k)})$ using PyTorch matrix operations.
* **Optimization:** Used Log-Softmax and Cross-Entropy Loss for numerical stability during training.
