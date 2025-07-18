from typing import Dict
import numpy as np
from vision_classifier.classifiers.base import Classifier
from vision_classifier.storage.base import Storage
from vision_classifier.similarity import metrics
from collections import Counter
from vision_classifier.registry import register_classifier

@register_classifier("knn")
class KNNClassifier(Classifier):
    def __init__(self, k=5, similarity_metric="cosine_similarity"):
        self.k = k
        self.similarity_metric_name = similarity_metric
        self.similarity_metric = getattr(metrics, similarity_metric)
        self.storage = None

    def fit(self, storage: Storage):
        self.storage = storage

    def predict(self, query_embedding: np.ndarray) -> Dict:
        if not self.storage:
            raise ValueError("Classifier has not been fitted. Call fit() first.")

        distances = []
        for class_name in self.storage.get_all_classes():
            for embedding in self.storage.get_embeddings(class_name):
                dist = self.similarity_metric(query_embedding, embedding)
                distances.append((dist, class_name))
        
        # Sort by distance (descending for similarity, ascending for distance)
        distances.sort(key=lambda x: x[0], reverse=True)
        
        # Get top k neighbors
        top_k_neighbors = distances[:self.k]
        
        # Majority vote
        class_votes = Counter([neighbor[1] for neighbor in top_k_neighbors])
        prediction, vote_count = class_votes.most_common(1)[0]
        
        confidence = vote_count / self.k
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "neighbors": top_k_neighbors
        }

    def get_config(self):
        return {"k": self.k, "similarity_metric": self.similarity_metric_name}

    def get_name(self):
        return f"knn-{self.k}"
