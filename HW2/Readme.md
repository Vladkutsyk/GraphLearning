# Graph Machine Learning: Node Classification Comparison

This repository contains the implementation and evaluation of four different models for node classification. The goal was to compare how graph structure, node features, and message aggregation strategies contribute to the model's performance.

## 📌 Project Overview

We compared four distinct approaches:
1.  **DeepWalk (Graph Only):** Learned embeddings based strictly on graph topology using random walks and Word2Vec.
2.  **MLP (Features Only):** A standard 2-layer neural network using only node features for classification.
3.  **Vanilla GNN (Graph + Features):** A custom 2-layer Graph Neural Network implementation aggregating local neighborhoods using Sum Aggregation.
4.  **Vanilla GNN with Mean Aggregation:** An improved GNN that row-normalizes the adjacency matrix ($\tilde{A} = D^{-1} \hat{A}$) to stabilize message passing and prevent high-degree nodes from dominating the embedding space.

## 📊 Results & Evaluation

The models were tested on the same dataset. Combining topology and features yielded significant improvements, but applying degree normalization (Mean Aggregation) provided the highest accuracy.

| Approach | Information Used | Test Accuracy |
| :--- | :--- | :---: |
| 1. DeepWalk | Graph Topology Only | 71.60% |
| 2. MLP | Node Features Only | 53.40% |
| 3. Vanilla GNN | Graph + Features | 75.10% |
| **4. Vanilla GNN (Mean Agg)**| **Graph + Features** | **80.00%** |

## 🌌 Visualizations (t-SNE)

We visualized the node embeddings using t-SNE to compare how well each model clusters different classes:

<img width="1489" height="495" alt="image" src="https://github.com/user-attachments/assets/0ce7ed58-3f02-49b6-848e-37ec214b89a6" />


* **DeepWalk:** Clusters nodes based on graph proximity. Shows reasonable separation but lacks feature awareness.
* **MLP:** Embeddings are scattered. Without structural context, features alone are insufficient for clear class separation.
* **Vanilla GNN (Sum Aggregation):** Shows distinct and clean color clusters. By performing message passing, the GNN successfully groups nodes of the same class.
* **Vanilla GNN (Mean Aggregation):** Yields the clearest class separation. By normalizing the messages by node degree, it prevents feature explosion for popular nodes, resulting in tighter and more accurately defined clusters.

## 🧠 Key Technical Implementations

* **Custom Adjacency Builder:** Efficiently creates a dense adjacency matrix with self-loops from a PyG `edge_index`.
* **Degree Normalization:** Implemented a row-normalized adjacency builder to shift from sum aggregation to mean aggregation ($D^{-1} \hat{A}$), improving numerical stability.
* **Vanilla GNN Layer:** Implemented the message passing step $H^{(k+1)} = \tilde{A} \cdot (H^{(k)} W^{(k)})$ using PyTorch matrix operations.
* **Optimization:** Used Log-Softmax and Cross-Entropy Loss for numerical stability during training.
