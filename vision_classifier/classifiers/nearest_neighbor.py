from typing import Dict
import numpy as np
from vision_classifier.classifiers.base import Classifier
from vision_classifier.storage.base import Storage
from vision_classifier.similarity import metrics
from vision_classifier.registry import register_classifier

@register_classifier("nearest_neighbor")
class NearestNeighborClassifier(Classifier):
    def __init__(self, similarity_metric="cosine_similarity", threshold=0.75):
        self.similarity_metric_name = similarity_metric
        self.similarity_metric = getattr(metrics, similarity_metric)
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

    def get_config(self):
        return {"similarity_metric": self.similarity_metric_name, "threshold": self.threshold}

    def get_name(self):
        return "nearest_neighbor"
