import numpy as np

def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings."""
    return np.dot(emb1.flatten(), emb2.flatten()) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def euclidean_distance(emb1, emb2):
    """Calculate euclidean distance between two embeddings."""
    return np.linalg.norm(emb1 - emb2)
