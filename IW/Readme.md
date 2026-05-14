# 🗳️ Wiki-RfA: Edge Sign Prediction with Graph Neural Networks + NLP

> Predicting **trust vs. distrust** in Wikipedia's admin elections using signed graph neural networks and sentence-level text embeddings.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-red)](https://pyg.org)
[![Dataset](https://img.shields.io/badge/Dataset-SNAP_Wiki--RfA-green)](https://snap.stanford.edu/data/wiki-RfA.html)

---

## 📋 Table of Contents

1. [The Dataset](#1-the-dataset)
2. [Graph Construction](#2-graph-construction)
3. [Task Definition](#3-task-definition)
4. [Train / Test Split](#4-train--test-split)
5. [Feature Engineering & Edge Messages](#5-feature-engineering--edge-messages)
6. [Model Architectures](#6-model-architectures)
7. [Evaluation Metrics](#7-evaluation-metrics)

---

## 1. The Dataset

**Source**: [SNAP Stanford — Wiki-RfA](https://snap.stanford.edu/data/wiki-RfA.html)

Wikipedia requires a public community vote before granting a user administrator privileges. These elections are called **Requests for Adminship (RfA)**. Any community member can vote on any nomination, leaving a written comment alongside their ballot. The Wiki-RfA dataset captures every recorded vote from the English Wikipedia RfA process.

### 1.1 Record Format

Each vote is stored as a key-value block in plain text:

```
SRC: Steel1943          ← voter's username
TGT: BDD                ← candidate's username
VOT: 1                  ← vote: -1=oppose, 0=neutral, +1=support
RES: 1                  ← election outcome: -1=rejected, +1=accepted
YEA: 2013               ← year the election was started
DAT: 23:13, 19 Apr 2013 ← exact timestamp
TXT: '''Support''' as co-nom.   ← free-text comment in wiki markup
```

### 1.2 Attribute Schema

| Field | Type | Description |
|-------|------|-------------|
| `SRC` | string | Username of the voter — becomes a **node** |
| `TGT` | string | Username of the candidate — becomes a **node** |
| `VOT` | int {-1, 0, +1} | Vote value — becomes the **edge label** |
| `RES` | int {-1, +1} | Final election result — **node label** for candidates |
| `YEA` | int | Year election was opened |
| `DAT` | string | Precise vote timestamp |
| `TXT` | string | Written justification in wiki markup — used as **edge feature** |

### 1.3 Key Statistics

| Statistic | Value |
|-----------|-------|
| Total vote records (edges) | ~159,388 |
| Unique voters (SRC) | ~7,118 |
| Unique candidates (TGT) | ~2,794 |
| **Total unique users (nodes)** | **~10,835** |
| Year range | 2003 – 2013 |
| Support votes (+1) | ~73.2 % |
| Oppose votes  (−1) | ~25.6 % |
| Neutral votes (0)  | ~1.2 %  |
| Accepted candidates | ~61.3 % |
| Rejected candidates | ~38.7 % |

### 1.4 Collectible Statistics & Insights

Working with this dataset allows us to compute and observe:

- **Degree distributions** — both in-degree (votes received as candidate) and out-degree (votes cast as voter) follow approximate power laws, indicating a small number of highly active users
- **Temporal trends** — the ratio of support vs. oppose votes shifted over the years as community standards evolved
- **Reciprocity** — the fraction of mutual voting pairs reveals whether voters tend to return votes
- **Clustering coefficient** — how tightly knit voter sub-communities are
- **Balance theory compliance** — whether the signed graph follows sociological balance: "friend of my friend is my friend", "enemy of my enemy is my friend"
- **Election predictability** — candidates with high early support ratios almost always win, making vote sign a strong predictor of outcome

---

## 2. Graph Construction

The raw text data is parsed into a **directed, signed, attributed graph**.

### 2.1 Structure

```
       VOT = +1  (support)
SRC ─────────────────────► TGT
(voter node)                (candidate node)
       VOT = -1  (oppose)
SRC ─────────────────────► TGT
```

- **Nodes** = all unique Wikipedia usernames (both voters and candidates)
- **Directed edges** = one vote cast by SRC toward TGT
- **Edge sign** = VOT ∈ {−1, +1} (neutral votes are dropped, < 1 %)
- **Edge attribute** = TXT comment text, encoded as a 384-dim sentence embedding
- **Node type** is implicit: some users appear only as voters, some only as candidates, and many as both

### 2.2 Graph Schema Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Wiki-RfA Graph                             │
│                                                                 │
│   Node (user)                                                   │
│   ┌──────────────┐                                              │
│   │  username    │  features: [out_deg, in_deg,                 │
│   │              │             pos_given, neg_given,            │
│   │              │             pos_recv,  neg_recv,             │
│   │              │             is_candidate]                    │
│   └──────┬───────┘                                              │
│          │                                                      │
│          │  Directed edge (vote)                                │
│          │  ┌──────────────────────────────────────────┐        │
│          └─►│  sign: +1 (support) / -1 (oppose)        │        │
│             │  text: sentence embedding (384 dim)      │        │
│             └──────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Signed Subgraph

For the SGCN model, the edge set is split into two separate adjacency structures:

```
Full graph G  ──┬──► G⁺  (positive edges, VOT = +1)  ~73 k edges
                │
                └──► G⁻  (negative edges, VOT = -1)  ~26 k edges
```

Both sub-graphs are made **bidirectional** for message passing (each directed edge gets a reverse copy), so the GNN can aggregate information from both directions.

### 2.4 Subgraph Visualisation (schematic)

```
        [Voter A] ──+1──► [Candidate X] ◄──-1── [Voter B]
                                │
                               +1
                                │
                                ▼
        [Voter C] ──+1──► [Candidate Y] ◄──+1── [Voter D]
                                │
                               -1
                                │
                                ▼
                          [Voter E] ──-1──► [Candidate Z]
```

Arrows represent directed votes; `+1` = green (support), `-1` = red (oppose).

---

## 3. Task Definition

### 3.1 Chosen Task: Edge Sign Prediction

Given the graph structure and node/edge features, **predict the sign of each vote edge**:

```
Input:   Graph topology G, node features x_u, x_v,
         edge text embedding t_{uv}
Output:  ŷ ∈ {0, 1}   where  0 = Oppose (VOT=-1)
                               1 = Support (VOT=+1)
```

This is a **binary edge classification** problem.

### 3.2 Why This Task?

| Alternative task | Why not chosen |
|---|---|
| Node classification (predict RES) | Only ~2,794 candidates have labels; not enough supervision |
| Graph classification | Would require multiple separate graphs; doesn't apply here |
| Link prediction | Topology is already fully observed |
| **Edge sign prediction** ✅ | ~159k labeled edges, natural signed structure, benchmark task |

### 3.3 Task Format

```
Training signal:  Edge (SRC_i, TGT_i) with known label VOT_i ∈ {-1, +1}
                  → mapped to binary: 0 (oppose) / 1 (support)

Prediction:       For a new edge (u, v), compute
                  P(support | h_u, h_v, text_{uv})

Loss:             Weighted Cross-Entropy
                  (weight on positive class to handle 73/27 imbalance)
```

---

## 4. Train / Test Split

### 4.1 Temporal Split — Preventing Data Leakage

Edges are sorted chronologically by `YEA` (year). The first **80 %** become the training set and the last **20 %** become the test set.

```
Timeline ─────────────────────────────────────────────────────────►
2003  2004  2005  2006  2007  2008  2009  2010  2011  2012  2013
│◄───────────────────── TRAIN (80%) ──────────────────────►│◄─TEST─►│
```

| Split | Size | Years |
|-------|------|-------|
| Train | ~127,000 edges | 2003 – ~2011 |
| Test  | ~32,000 edges  | ~2011 – 2013 |

### 4.2 Why Temporal, Not Random?

A **random split** would create data leakage in three ways:

```
Problem 1 — Node feature leakage
  If test edges are used to compute degree or vote-ratio features,
  the model sees future information about users.
  → We compute ALL node features exclusively from TRAIN edges.

Problem 2 — Graph structure leakage
  If test edges are included in the message-passing adjacency,
  the model indirectly observes future votes during node embedding.
  → Message passing uses ONLY train-split edges.

Problem 3 — Temporal concept drift
  Wikipedia community norms shifted over time. A random split
  ignores this, making train and test artificially similar.
  → Temporal split tests real generalization to the future.
```

### 4.3 Leakage-Free Feature Computation

```python
# ✅ CORRECT — features computed from train indices only
np.add.at(F_mat[:, 0], tr_s_arr, 1)          # out-degree
np.add.at(F_mat[:, 2], tr_s_arr[pos_train], 1)  # positive votes given

# ✅ CORRECT — message passing graph uses train edges only
tr_ei_bi  = bidirectional(train_edges)
ei_pos_bi = bidirectional(train_positive_edges)
ei_neg_bi = bidirectional(train_negative_edges)

# ❌ WRONG (not done) — never include test edges in message passing
```

---

## 5. Feature Engineering & Edge Messages

### 5.1 Node Features (7 structural features)

Node features are computed strictly from the **training split** and then standardised with `StandardScaler`:

| # | Feature | Description |
|---|---------|-------------|
| 0 | `out_degree` | Number of votes cast by this user (train only) |
| 1 | `in_degree` | Number of votes received as candidate (train only) |
| 2 | `pos_given` | Positive (+1) votes cast (train only) |
| 3 | `neg_given` | Negative (−1) votes cast (train only) |
| 4 | `pos_received` | Positive (+1) votes received (train only) |
| 5 | `neg_received` | Negative (−1) votes received (train only) |
| 6 | `is_candidate` | 1 if user ever ran for admin, 0 otherwise |

```
Node feature vector x_u ∈ ℝ⁷  (after StandardScaler)

x_u = [out_deg, in_deg, pos_given, neg_given, pos_recv, neg_recv, is_cand]
         ↑         ↑        ↑           ↑          ↑          ↑       ↑
       activity  fame    trust_out  distrust_out  credibility hostility  role
```

### 5.2 Edge Text Features (NLP)

Each vote comes with a written comment. We convert this into a dense vector using a **Sentence Transformer**.

**Model used**: `all-MiniLM-L6-v2` (384-dimensional embeddings, fast and lightweight)

#### Text Cleaning — Preventing Label Leakage

Raw comments almost always begin with an explicit keyword:
```
'''Support''' as co-nominator.
'''Oppose''' per WP:ADMIN — not enough edits.
'''Neutral''' — I don't know this user.
```

If we feed these raw texts to the model, it would simply learn to detect the words "Support" / "Oppose" — a trivial shortcut that reveals the answer. To prevent this:

```python
# Regex strips voting keywords from the START of the comment only
pattern = r"(?i)^[\'\"*\s]*(strong |weak )?(support|oppose|neutral)[\'\"*\s\-:.|,;]*"

# Before cleaning:  "'''Support''' as co-nominator."
# After  cleaning:  "as co-nominator."
```

This means the NLP model is forced to read **the reasoning**, not the verdict.

#### Pipeline

```
Raw TXT comment
      │
      ▼
  Text Cleaning  →  strip "Support/Oppose/Neutral" from beginning
      │
      ▼
  SentenceTransformer('all-MiniLM-L6-v2')
      │
      ▼
  t_{uv}  ∈  ℝ³⁸⁴   (sentence embedding)
      │
      ▼
  Linear projection  →  t_proj ∈ ℝʰⁱᵈ   (compressed to model hidden dim)
      │
      ▼
  Concatenated into EdgePredictor alongside GNN node embeddings
```

### 5.3 How Text Is Used in the EdgePredictor

For a directed edge `(u → v)` with comment `t`:

```
                    ┌─ h_u  (GNN embedding of voter)      ──────────┐
                    │                                               │
input to predictor: ├─ h_v  (GNN embedding of candidate)  ──────────┤ → concat → MLP → logits
                    │                                               │
                    └─ t_proj  (projected text embedding) ──────────┘

For GCN/SAGE/GAT:   input_dim = hid + hid + hid  = 3·hid
For SGCN:           input_dim = hid + hid + hid + hid + hid = 5·hid
                                  h⁺_u  h⁻_u  h⁺_v  h⁻_v  t_proj
```

This fusion means every prediction uses:
- **Graph context** (how the community is connected)
- **User reputation** (what the voter/candidate's history looks like)
- **Vote reasoning** (what the commenter actually wrote)

---

## 6. Model Architectures

All four models share the same **Encoder → EdgePredictor** paradigm:

```
┌──────────────────────────────────────────────────────────────────┐
│                   General Pipeline                               │
│                                                                  │
│  Node features x  ──► GNN Encoder ──► node embeddings h          │
│                                                                  │
│  Edge (u,v):  [h_u ‖ h_v ‖ t_proj]  ──► MLP ──► logits ──► ŷ     │
│                                                                  │
│  (where t_proj = Linear(sentence_embedding))                     │
└──────────────────────────────────────────────────────────────────┘
```

### 6.1 GCN — Graph Convolutional Network

**Reference**: Kipf & Welling, *Semi-Supervised Classification with GCNs*, ICLR 2017

#### How It Works

GCN performs a **symmetric, spectral-domain** aggregation. Each layer updates node `i` by averaging its neighbours' features, normalised by their degrees:

```
h_i^(l+1) = ReLU( BatchNorm( Σ_{j ∈ N(i)∪{i}}  (1/√d_i·d_j) · W · h_j^(l) ) )
```

Where `d_i` is the degree of node `i`.

#### Architecture Diagram

```
x ∈ ℝ⁷  (node features)
  │
  ▼
GCNConv(7 → hid)  +  BatchNorm  +  ReLU  +  Dropout
  │
  ▼
GCNConv(hid → hid)  +  BatchNorm  +  ReLU  +  Dropout
  │
  ▼  [repeat × n_layers]
  │
  ▼
h ∈ ℝʰⁱᵈ  per node
  │
  ▼
EdgePredictor( [h_u ‖ h_v ‖ t_proj] → hid → 2 )
  │
  ▼
logits ∈ ℝ²  →  softmax  →  P(oppose), P(support)
```

**Limitation**: GCN treats all edges equally and ignores their signs. It uses the full (unsigned) bidirectional training graph for message passing.

---

### 6.2 GraphSAGE — Inductive Representation Learning

**Reference**: Hamilton et al., *Inductive Representation Learning on Large Graphs*, NeurIPS 2017

#### How It Works

GraphSAGE **samples and aggregates** neighbourhood features. The key difference from GCN: it **concatenates the node's own embedding** with the aggregated neighbourhood mean, enabling inductive generalisation:

```
h_N(i)^(l) = MEAN( { h_j^(l) : j ∈ N(i) } )

h_i^(l+1) = ReLU( BatchNorm( W · [h_i^(l) ‖ h_N(i)^(l)] ) )
```

#### Architecture Diagram

```
x ∈ ℝ⁷
  │
  ▼
SAGEConv(7 → hid, aggr='mean')  +  BatchNorm  +  ReLU  +  Dropout
  │                   │
  │               mean of
  │               neighbours
  ▼
SAGEConv(hid → hid)  +  ...
  │
  ▼
h ∈ ℝʰⁱᵈ  per node
  │
  ▼
EdgePredictor( [h_u ‖ h_v ‖ t_proj] → hid → 2 )
```

**Advantage over GCN**: naturally handles unseen nodes; no need to recompute the full graph Laplacian.

---

### 6.3 GAT — Graph Attention Network

**Reference**: Veličković et al., *Graph Attention Networks*, ICLR 2018

#### How It Works

Instead of fixed normalisation, GAT learns **attention weights** α_{ij} for each edge, telling the model which neighbours to weight more heavily:

```
α_{ij} = softmax_j( LeakyReLU( a^T · [W·h_i ‖ W·h_j] ) )

h_i^(l+1) = ELU( BatchNorm( Σ_{j ∈ N(i)} α_{ij} · W · h_j ) )
```

Multi-head attention (`heads=4` or `heads=8`) runs multiple independent attention mechanisms in parallel, then concatenates the results.

#### Architecture Diagram

```
x ∈ ℝ⁷
  │
  ▼
GATConv(7 → head_dim, heads=H, concat=True)
  │  ↑
  │  Attention scores α_{ij} learned per edge
  │
  ▼  (output: hid = head_dim × H)
BatchNorm  +  ELU  +  Dropout
  │
  ▼
GATConv(hid → head_dim, heads=H)  +  ...
  │
  ▼
h ∈ ℝʰⁱᵈ  per node
  │
  ▼
EdgePredictor( [h_u ‖ h_v ‖ t_proj] → hid → 2 )
```

**Advantage**: can focus on the most relevant voters/candidates in a neighbourhood, ignoring noisy or irrelevant connections.

---

### 6.4 Improved SGCN — Signed Graph Convolutional Network

**Reference**: Derr et al., *Signed Graph Convolutional Network*, ICDM 2018  
**Our improvement**: added **skip connections** and replaced `ReLU` with `ELU` for better gradient flow.

#### The Core Idea: Balance Theory

Social balance theory (Heider, 1946) defines rules for how signed relationships propagate:

```
 (+) × (+) = (+)    "Friend of my friend is my friend"
 (−) × (−) = (+)    "Enemy of my enemy is my friend"
 (+) × (−) = (−)    "Friend of my enemy is my enemy"
 (−) × (+) = (−)    "Enemy of my friend is my enemy"
```

#### Dual Embeddings

Every node maintains **two separate embeddings**:

```
h⁺_i  ∈  ℝʰⁱᵈ  — "positive/friendship" representation
h⁻_i  ∈  ℝʰⁱᵈ  — "negative/hostility"  representation
```

#### Update Rules (ImprovedSGCNLayer)

```
Positive update:
  h⁺_i^(l+1) = ELU( BN(
      W_pp · MEAN_{j ∈ N⁺(i)}(h⁺_j)     ← friend's friend = friend
    + W_nn · MEAN_{j ∈ N⁻(i)}(h⁻_j)     ← enemy's enemy   = friend
    + skip_p · h⁺_i                       ← skip connection (NEW)
  ))

Negative update:
  h⁻_i^(l+1) = ELU( BN(
      W_pn · MEAN_{j ∈ N⁺(i)}(h⁻_j)     ← friend's enemy  = enemy
    + W_np · MEAN_{j ∈ N⁻(i)}(h⁺_j)     ← enemy's friend  = enemy
    + skip_n · h⁻_i                       ← skip connection (NEW)
  ))
```

#### Architecture Diagram

```
x ∈ ℝ⁷
  │
  ├── proj_p ──► h⁺ ∈ ℝʰⁱᵈ   (initial positive embedding)
  └── proj_n ──► h⁻ ∈ ℝʰⁱᵈ   (initial negative embedding)
         │
         ▼  ×n_layers
  ┌─────────────────────────────────────┐
  │  ImprovedSGCNLayer                  │
  │                                     │
  │  h⁺ ◄── W_pp·AGG(G⁺, h⁺)            │
  │       +  W_nn·AGG(G⁻, h⁻)           │
  │       +  skip_p·h⁺   (residual)     │
  │                                     │
  │  h⁻ ◄── W_pn·AGG(G⁺, h⁻)            │
  │       +  W_np·AGG(G⁻, h⁺)           │
  │       +  skip_n·h⁻   (residual)     │
  └─────────────────────────────────────┘
         │
         ▼
  h⁺_u, h⁻_u, h⁺_v, h⁻_v  ∈  ℝʰⁱᵈ  each
         │
         ▼
  EdgePredictor( [h⁺_u ‖ h⁻_u ‖ h⁺_v ‖ h⁻_v ‖ t_proj] → 2·hid → 2 )
         │
         ▼
  logits → P(oppose), P(support)
```

#### Why the Skip Connection Matters

Without residual connections, after multiple SGCN layers some nodes (e.g. those with **no** positive or **no** negative neighbours) receive only zero aggregations and their embeddings collapse. The skip connection ensures that even isolated-in-one-sign nodes retain their identity:

```
Standard SGCN:  h⁺_new = ELU( BN( W_pp·agg_pp + W_nn·agg_nn ) )
                                    ↑ could be 0   ↑ could be 0  → gradient death

Improved SGCN:  h⁺_new = ELU( BN( W_pp·agg_pp + W_nn·agg_nn + skip·h⁺_old ) )
                                                                    ↑ always non-zero
```

### 6.5 Model Comparison Summary

| Model | Uses sign | Message passing | Text | Skip conn. | Input to predictor |
|-------|-----------|----------------|------|------------|-------------------|
| GCN | ✗ | Spectral (normalised) | ✅ | — | `h_u ‖ h_v ‖ t` |
| GraphSAGE | ✗ | Inductive mean | ✅ | — | `h_u ‖ h_v ‖ t` |
| GAT | ✗ | Attention-weighted | ✅ | — | `h_u ‖ h_v ‖ t` |
| **SGCN** | **✅** | **Balance Theory** | **✅** | **✅** | **`h⁺_u ‖ h⁻_u ‖ h⁺_v ‖ h⁻_v ‖ t`** |

---

## 7. Evaluation Metrics

We evaluate all models using **four primary metrics**. Simple accuracy is deliberately avoided as the primary criterion — here's why.

### 7.1 Why Accuracy Alone Is Misleading

The dataset has a **73 % support / 27 % oppose** class imbalance. A model that predicts "Support" for every single edge would achieve:

```
Accuracy = 73 %  ← looks reasonable!
```

But this model is completely useless — it never detects a single oppose vote. Accuracy cannot distinguish this trivial baseline from a genuinely useful model. The metrics below are designed to expose exactly this kind of failure.

---

### 7.2 AUC-ROC — Area Under the ROC Curve

**What it measures**: the model's ability to **rank** support votes above oppose votes across all possible decision thresholds.

**Formula**:
```
ROC curve plots TPR vs FPR at every threshold τ ∈ [0, 1]:

  TPR(τ) = TP / (TP + FN)    ← fraction of support votes correctly identified
  FPR(τ) = FP / (FP + TN)    ← fraction of oppose votes incorrectly labelled support

AUC-ROC = ∫₀¹ TPR(FPR) d(FPR)
```

**Interpretation**:
- `AUC = 1.0` → perfect separation
- `AUC = 0.5` → no better than random coin flip
- `AUC = 0.73` → the model correctly ranks a random support edge above a random oppose edge 73 % of the time

**Why important here**: captures overall discriminative power regardless of threshold choice. Robust to class imbalance.

---

### 7.3 Average Precision (AP) — Area Under the PR Curve

**What it measures**: quality of predictions specifically for the **minority class (oppose)**, where false positives matter most.

**Formula**:
```
PR curve plots Precision vs Recall at every threshold:

  Precision(τ) = TP / (TP + FP)   ← of all predicted support, how many are really support?
  Recall(τ)    = TP / (TP + FN)   ← of all true support, how many did we find?

AP = Σ_n (Recall_n − Recall_{n-1}) · Precision_n
```

**Interpretation**: a random classifier achieves `AP ≈ 0.73` (equal to the positive class rate). Any model worth using must significantly exceed this baseline.

**Why important here**: with imbalanced classes, the PR curve reveals whether the model genuinely separates classes or just inherits the majority class frequency. It punishes excessive false positives much harder than ROC does.

---

### 7.4 F1-Score

**What it measures**: the **harmonic mean of Precision and Recall** — a single number that penalises both missing true opposers (low recall) and labelling supporters as opposers (low precision).

**Formula**:
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)

F1 = 2 · (Precision × Recall) / (Precision + Recall)
   = 2·TP / (2·TP + FP + FN)
```

**Interpretation**: `F1 = 1.0` is perfect; `F1 = 0.0` is worst. The harmonic mean ensures that a high score requires both precision and recall to be high simultaneously — you cannot compensate a terrible recall with perfect precision.

**Why important here**: directly measures whether the model is useful in practice — can you trust its predictions in both directions?

---

### 7.5 MCC — Matthews Correlation Coefficient

**What it measures**: the overall quality of a binary classification, **taking all four cells of the confusion matrix into account**.

**Formula**:
```
MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Interpretation**:
- `MCC = +1` → perfect prediction
- `MCC =  0` → no better than random
- `MCC = -1` → perfectly wrong (systematic inversion)

**Why the best single-number metric for imbalanced data**: MCC is mathematically equivalent to the Pearson correlation between predicted and actual labels. Unlike F1, it equally considers all four quadrants of the confusion matrix. A model that predicts everything as "Support" gets:

```
TP ≈ 127k,  TN = 0,  FP = 0,  FN ≈ 32k

MCC = (127k·0 − 0·32k) / √(...) = 0 / ... = 0
```

Zero — correctly identifying it as no better than chance.

---

### 7.6 Metric Summary Table

| Metric | Range | Imbalance-robust | Threshold-free | Uses all 4 CM cells |
|--------|-------|:-:|:-:|:-:|
| Accuracy | [0, 1] | ✗ | ✓ | ✗ |
| **AUC-ROC** | [0, 1] | ✓ | ✓ | ✗ |
| **Avg Precision** | [0, 1] | ✓ | ✓ | ✗ |
| **F1** | [0, 1] | ~✓ | ✗ | ✗ |
| **MCC** | [−1, +1] | ✓ | ✗ | ✓ |

All four chosen metrics together provide complementary views:
- **AUC** = how well does it rank?
- **AP** = how well does it handle the harder class?
- **F1** = is it practically useful at the chosen threshold?
- **MCC** = what does the full confusion matrix say?

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/wiki-rfa-gnn.git
cd wiki-rfa-gnn

# 2. Open in Google Colab (recommended — free GPU)
#    Upload wiki_rfa_gnn_upgraded_v4.ipynb and run all cells.
#    The notebook downloads the dataset automatically.

# 3. Or run locally
pip install torch torch_geometric sentence-transformers scikit-learn matplotlib seaborn networkx pandas
jupyter notebook wiki_rfa_gnn_upgraded_v4.ipynb
```

The notebook downloads the dataset automatically from SNAP Stanford on first run.

---

## File Structure

```
wiki-rfa-gnn/
├── wiki_rfa_gnn_upgraded_v4.ipynb   ← main notebook
├── README.md                         ← this file
└── (generated outputs)
```

---

## References

1. Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
2. Hamilton et al. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS.
3. Veličković et al. (2018). *Graph Attention Networks.* ICLR.
4. Derr et al. (2018). *Signed Graph Convolutional Network.* ICDM.
5. Leskovec et al. (2010). *Signed Networks in Social Media.* WWW. *(original Wiki-RfA paper)*
6. Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
