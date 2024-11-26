import networkx as nx
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Create a Citation Network
def create_citation_network(edges):
    """
    Create a citation graph from a list of edges.
    :param edges: List of tuples (paper1, paper2), where paper1 cites paper2.
    :return: NetworkX directed graph.
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G

# Example edge list (Paper A cites B, B cites C, etc.)
edges = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "D"), ("D", "A")]
G = create_citation_network(edges)

# Step 2: Generate Node Embeddings using Node2Vec
def generate_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, workers=4):
    """
    Generate node embeddings using Node2Vec.
    :param graph: NetworkX graph.
    :param dimensions: Number of dimensions for embeddings.
    :param walk_length: Length of random walks.
    :param num_walks: Number of random walks per node.
    :param workers: Number of workers for parallel processing.
    :return: Dictionary mapping nodes to embeddings.
    """
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length,
                        num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return {node: model.wv[node] for node in graph.nodes()}

embeddings = generate_embeddings(G)

# Step 3: Retrieve Similar Nodes
def retrieve_similar_nodes(target_node, embeddings, top_k=3):
    """
    Retrieve nodes most similar to the target node based on cosine similarity.
    :param target_node: Node to query.
    :param embeddings: Dictionary of node embeddings.
    :param top_k: Number of top similar nodes to retrieve.
    :return: List of top-k similar nodes.
    """
    if target_node not in embeddings:
        raise ValueError(f"Node {target_node} not found in embeddings.")
    
    target_vector = embeddings[target_node].reshape(1, -1)
    all_nodes = list(embeddings.keys())
    all_vectors = np.array([embeddings[node] for node in all_nodes])
    
    similarities = cosine_similarity(target_vector, all_vectors).flatten()
    similar_indices = similarities.argsort()[::-1][1:top_k + 1]
    return [(all_nodes[i], similarities[i]) for i in similar_indices]

# Query similar papers to a target node
target_node = "A"
similar_nodes = retrieve_similar_nodes(target_node, embeddings)
print(f"Top similar nodes to {target_node}: {similar_nodes}")
