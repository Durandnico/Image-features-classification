from typing import Dict
import numpy as np
from .base import Classifier
from ..storage.base import Storage
from ..similarity.metrics import cosine_similarity

class NearestNeighborClassifier(Classifier):
    def __init__(self, similarity_metric=cosine_similarity, threshold=0.75):
        self.similarity_metric = similarity_metric
        self.threshold = threshold
        self.storage = None

    def fit(self, storage: Storage):
        self.storage = storage

    def predict(self, query_embedding: np.ndarray) -> Dict:
        if not self.storage:
            raise ValueError("Classifier has not been fitted. Call fit() first.")

        results = {}
        for class_name in self.storage.get_all_classes():
            embeddings = self.storage.get_embeddings(class_name)
            similarities = [self.similarity_metric(query_embedding, emb) for emb in embeddings]
            if not similarities:
                results[class_name] = {
                    "max_similarity": -1,
                    "mean_similarity": -1,
                    "top_example_idx": -1
                }
                continue
            results[class_name] = {
                "max_similarity": max(similarities),
                "mean_similarity": sum(similarities) / len(similarities),
                "top_example_idx": similarities.index(max(similarities))
            }
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]["max_similarity"], reverse=True)
        
        prediction = sorted_results[0][0]
        confidence = sorted_results[0][1]["max_similarity"]
        
        if confidence < self.threshold:
            prediction = "unknown"
            
        return {
            "prediction": prediction,
            "confidence": confidence,
            "top_k": sorted_results
        }
