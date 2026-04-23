"""
student_module.py  –  STUDENT VERSION
======================================
Graph Machine Learning Lab: Shallow Node Embeddings with Random Walks

Your task is to implement every function that currently raises
NotImplementedError.  Read the docstring carefully; it describes
the expected inputs, outputs, and the algorithm to follow.

Tips
----
* Useful imports are already at the top of this file.
* You can test individual functions in the notebook before moving on.
* numpy.random.choice is your friend for sampling from a distribution.
* When stuck on the maths, re-read the relevant notebook section.
"""

import random
import numpy as np
import networkx as nx
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ===========================================================================
# SECTION 1 – DeepWalk utilities
# ===========================================================================

def uniform_random_walk(G: nx.Graph, start: int, length: int) -> list:
    """
    Perform a single uniform (unbiased) random walk on graph G.

    At each step, choose one of the current node's neighbours uniformly at
    random and append it to the walk.  Store node IDs as **strings** so that
    the walk can be fed directly into Word2Vec (every node is a "word").

    Algorithm
    ---------
    1. Initialise the walk as [str(start)].
    2. Repeat `length` times:
       a. Get the list of neighbours of the current node.
       b. If there are no neighbours, stop early.
       c. Sample one neighbour uniformly at random.
       d. Append str(sampled_neighbour) to the walk.
       e. Update the current node.
    3. Return the walk.

    Parameters
    ----------
    G      : NetworkX graph to walk on.
    start  : Starting node ID (integer).
    length : Number of steps (the walk will contain at most length+1 nodes).

    Returns
    -------
    walk : list of str, e.g. ['0', '3', '7', ...]

    Example
    -------
    >>> G = nx.erdos_renyi_graph(10, 0.4, seed=1)
    >>> walk = uniform_random_walk(G, start=0, length=5)
    >>> len(walk)  # should be 6 (start + 5 steps)
    6
    >>> all(isinstance(n, str) for n in walk)  # node IDs must be strings
    True
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    """
    Perform a single uniform (unbiased) random walk on graph G using NumPy.
    """
    # 1. Initialise the walk as [str(start)].
    walk = [str(start)]
    current_node = start

    # 2. Repeat `length` times:
    for _ in range(length):
        # a. Get the list of neighbours of the current node.
        neighbors = list(G.neighbors(current_node))

        # b. If there are no neighbours, stop early.
        if not neighbors:
            return walk
            #break

        # c. Sample one neighbour uniformly at random.
        # Since it is uniform, we don't need to pass the 'p' parameter.
        # numpy.random.choice picks an element from the array-like 'neighbors'.
        sampled_neighbour = random.choice(neighbors)

        # d. Append str(sampled_neighbour) to the walk.
        walk.append(str(sampled_neighbour))

        # e. Update the current node.
        current_node = sampled_neighbour

    # 3. Return the walk.
    return walk
    #raise NotImplementedError("Implement uniform_random_walk")
    # --------------------------------------------------------------------- #


def generate_walks(G: nx.Graph, num_walks: int, walk_length: int) -> list:
    """
    Generate a corpus of random walks over **all** nodes in G.

    For each training epoch (out of `num_walks`), shuffle the node order and
    start one walk from every node.  Shuffling prevents the model from
    seeing nodes in a fixed order every epoch.

    Parameters
    ----------
    G           : NetworkX graph.
    num_walks   : Number of walks to start from each node.
    walk_length : Number of steps per walk.

    Returns
    -------
    walks : list of lists of str  (a "corpus" for Word2Vec)

    Example
    -------
    >>> G = nx.karate_club_graph()
    >>> walks = generate_walks(G, num_walks=10, walk_length=5)
    >>> len(walks) == G.number_of_nodes() * 10
    True
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    walks = []
    # Get a list of all node IDs from the graph
    nodes = list(G.nodes())

    # 1. Repeat for each training epoch (num_walks)
    for _ in range(num_walks):
        
        # 2. Shuffle the node order to prevent ordering bias during training.
        # np.random.shuffle operates in-place on the list.
        random.shuffle(nodes)

        # 3. Start one walk from every node in the shuffled list
        for node in nodes:
            # Reusing the uniform_random_walk function implemented previously
            walk = uniform_random_walk(G, start=node, length=walk_length)
            
            # 4. Append the resulting walk (list of strings) to our corpus
            walks.append(walk)

    # 5. Return the full corpus (list of lists)
    return walks
    #raise NotImplementedError("Implement generate_walks")
    # --------------------------------------------------------------------- #


def train_embedding(
    walks: list,
    vector_size: int = 128,
    window: int = 10,
    epochs: int = 30,
    seed: int = 0,
) -> Word2Vec:
    """
    Train a skip-gram Word2Vec model on a corpus of random walks.

    Each walk is a sentence; each node ID is a word.  Use hierarchical
    softmax (hs=1) – this is what the original DeepWalk paper does.

    Parameters
    ----------
    walks       : Corpus produced by generate_walks().
    vector_size : Dimensionality of the node embedding vectors.
    window      : Context window size for skip-gram.
    epochs      : Number of training epochs.
    seed        : Random seed for reproducibility.

    Returns
    -------
    model : trained gensim Word2Vec instance.

    Hints
    -----
    * Create the model with Word2Vec(walks, hs=1, sg=1, ...).
    * Don't forget to build the vocabulary before training: model.build_vocab(walks).
    * Then call model.train(walks, total_examples=model.corpus_count,
      epochs=epochs).
    * Set workers=1 to keep results deterministic.
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    # 1. Create the model instance. 
    # We initialize it without the walks first to follow the multi-step hints.
    # min_count=0 ensures that every node in our walks gets an embedding, 
    # even if the node appears very infrequently.
    model = Word2Vec(
        walks,
        vector_size=vector_size,
        window=window,
        hs=1,        # Hierarchical Softmax (as per DeepWalk paper)
        sg=1,        # Skip-gram architecture
        seed=seed,
        workers=1,   # Ensures determinism
        min_count=0  # Don't drop any "rare" nodes
    )

    # 2. Build the vocabulary from the corpus of walks.
    # This maps every unique node ID string to an index in the embedding matrix.
    model.build_vocab(walks)

    # 3. Train the model.
    # total_examples=model.corpus_count tells Gensim how many walks to expect.
    model.train(
        walks, 
        total_examples=model.corpus_count, 
        epochs=epochs
    )

    return model
    #raise NotImplementedError("Implement train_embedding")
    # --------------------------------------------------------------------- #


def train_classifier(
    model: Word2Vec,
    labels: np.ndarray,
    train_mask: list,
    seed: int = 0,
) -> RandomForestClassifier:
    """
    Fit a Random Forest classifier on node embeddings.

    Use model.wv[train_mask] to retrieve the embedding matrix for the
    training nodes.

    Important: `train_mask` must contain node IDs as **strings** (e.g.
    ['0', '2', '4']), because walks store node IDs as text.  Use
    [int(n) for n in train_mask] to convert back to integers when
    indexing into the `labels` array.

    Parameters
    ----------
    model      : Trained Word2Vec model.
    labels     : Array of ground-truth node labels (integers).
    train_mask : List of node IDs as strings.
    seed       : Random seed for the classifier.

    Returns
    -------
    clf : Fitted RandomForestClassifier.
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    # 1. Retrieve the embedding matrix for the training nodes.
    # Gensim's .wv allows us to pass a list of strings directly.
    X_train = model.wv[train_mask]

    # 2. Convert string IDs to integers to index into the ground-truth labels.
    train_indices = [int(n) for n in train_mask]
    y_train = labels[train_indices]

    # 3. Initialize and fit the classifier.
    clf = RandomForestClassifier(random_state=seed)
    clf.fit(X_train, y_train)

    return clf
    #raise NotImplementedError("Implement train_classifier")
    # --------------------------------------------------------------------- #


def evaluate_classifier(
    clf: RandomForestClassifier,
    model: Word2Vec,
    labels: np.ndarray,
    test_mask: list,
) -> float:
    """
    Evaluate a fitted classifier on held-out nodes.

    Parameters
    ----------
    clf       : Fitted classifier.
    model     : Trained Word2Vec model.
    labels    : Full array of ground-truth node labels.
    test_mask : List of node IDs as **strings**.

    Returns
    -------
    acc : Float accuracy in [0, 1].
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    # 1. Retrieve the embedding matrix for the test nodes.
    X_test = model.wv[test_mask]

    # 2. Convert string IDs to integers for label indexing.
    test_indices = [int(n) for n in test_mask]
    y_test = labels[test_indices]

    # 3. Calculate accuracy. 
    # score() returns the mean accuracy on the given test data and labels.
    acc = clf.score(X_test, y_test)

    return acc
    #raise NotImplementedError("Implement evaluate_classifier")
    # --------------------------------------------------------------------- #


# ===========================================================================
# SECTION 2 – Node2Vec utilities (biased random walks)
# ===========================================================================

def biased_next_node(
    G: nx.Graph,
    previous,          # int or None
    current: int,
    p: float,
    q: float,
) -> int:
    """
    Sample the next node for a Node2Vec walk.

    Node2Vec defines a **second-order** random walk: the transition
    probability from `current` to each neighbour depends on the distance
    between that neighbour and the **previous** node.

    For each neighbour `v` of `current`, compute an unnormalised weight
    alpha(v) according to:

        alpha(v) = 1/p   if v == previous            (return step)
        alpha(v) = 1     if (v, previous) is an edge  (same distance)
        alpha(v) = 1/q   otherwise                    (explore farther)

    Then normalise the alphas to get a valid probability distribution and
    sample one neighbour.

    When `previous` is None (first step), fall back to a **uniform**
    transition (no bias yet).

    Parameters
    ----------
    G        : NetworkX graph.
    previous : Previously visited node (int), or None for the first step.
    current  : Current node (int).
    p        : Return parameter.  High p  -> less likely to backtrack.
    q        : In-out parameter.  Low q  -> BFS-like (local).
                                  High q -> DFS-like (exploratory).

    Returns
    -------
    next_node : int, the sampled next node.
    """
    neighbors = list(G.neighbors(current))
    
    # Handle the case where the node has no neighbors (sink node)
    if not neighbors:
        return None

    # 1. Fallback to a uniform transition if this is the first step (no 'previous' node)
    if previous is None:
        return random.choice(neighbors)

    # 2. Compute unnormalised weights (alphas) for each neighbor
    alphas = []
    for v in neighbors:
        if v == previous:
            # Case: Distance d_tx = 0 (Going back to where we came from)
            alphas.append(1.0 / p)
        elif G.has_edge(v, previous):
            # Case: Distance d_tx = 1 (Neighbor of both current and previous)
            alphas.append(1.0)
        else:
            # Case: Distance d_tx = 2 (New territory, not connected to previous)
            alphas.append(1.0 / q)

    # 3. Normalise the alphas to get a valid probability distribution
    # This turns the weights (e.g., [0.5, 1.0, 2.0]) into probabilities (sum = 1.0)
    total_weight = sum(alphas)
    probabilities = [a / total_weight for a in alphas]

    # 4. Sample the next node based on the distribution
    next_node = random.choice(neighbors, p=probabilities)

    return int(next_node)

    #if previous is None:
        # ---- YOUR CODE HERE (first step, uniform transition) ------------ #
        #raise NotImplementedError("Implement first-step fallback in biased_next_node")
        # ----------------------------------------------------------------- #

    # ---- YOUR CODE HERE (biased transition) ----------------------------- #
    #raise NotImplementedError("Implement biased transition in biased_next_node")
    # --------------------------------------------------------------------- #


def biased_random_walk(
    G: nx.Graph,
    start: int,
    length: int,
    p: float = 1.0,
    q: float = 1.0,
) -> list:
    """
    Perform a single second-order biased random walk (Node2Vec).

    At each step, call biased_next_node to decide where to go next,
    making sure to pass the correct `previous` node.

    Parameters
    ----------
    G      : NetworkX graph.
    start  : Starting node (int).
    length : Number of steps.
    p      : Return parameter.
    q      : In-out parameter.

    Returns
    -------
    walk : list of str node IDs.
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    walk = [str(start)]
    
    previous = None
    current = start

    # 2. Perform the walk for the specified length
    for _ in range(length):
        # Call our previously implemented biased_next_node
        # This function handles the logic for p and q
        next_node = biased_next_node(G, previous, current, p, q)

        # If we hit a dead end (sink node), stop the walk early
        if next_node is None:
            break

        # 3. Update the walk and the state
        walk.append(str(next_node))
        previous = current
        current = next_node

    return walk
    #raise NotImplementedError("Implement biased_random_walk")
    # --------------------------------------------------------------------- #


def generate_biased_walks(
    G: nx.Graph,
    num_walks: int,
    walk_length: int,
    p: float = 1.0,
    q: float = 1.0,
) -> list:
    """
    Generate a corpus of Node2Vec biased random walks over all nodes in G.

    Same shuffling logic as generate_walks, but call biased_random_walk
    instead of uniform_random_walk.

    Parameters
    ----------
    G           : NetworkX graph.
    num_walks   : Walks per node.
    walk_length : Steps per walk.
    p           : Return parameter.
    q           : In-out parameter.

    Returns
    -------
    walks : list of lists of str.
    """
    # ---- YOUR CODE HERE ------------------------------------------------- #
    walks = []
    nodes = list(G.nodes())

    # 1. Repeat the process for the number of walks (epochs)
    for _ in range(num_walks):
        # 2. Shuffle nodes to ensure the model doesn't learn ordering bias
        random.shuffle(nodes)

        # 3. Start a biased walk from every node in the graph
        for node in nodes:
            walk = biased_random_walk(
                G, 
                start=node, 
                length=walk_length, 
                p=p, 
                q=q
            )
            walks.append(walk)

    return walks
    #raise NotImplementedError("Implement generate_biased_walks")
    # --------------------------------------------------------------------- #
