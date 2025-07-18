from typing import Dict
import numpy as np
from vision_classifier.classifiers.base import Classifier
from vision_classifier.storage.base import Storage
from vision_classifier.similarity import metrics
from vision_classifier.registry import register_classifier

@register_classifier("nearest_centroid")
class NearestCentroidClassifier(Classifier):
    def __init__(self, similarity_metric="cosine_similarity"):
        self.similarity_metric_name = similarity_metric
        self.similarity_metric = getattr(metrics, similarity_metric)
        self.centroids = {}

    def fit(self, storage: Storage):
        self.storage = storage
        for class_name in self.storage.get_all_classes():
            embeddings = self.storage.get_embeddings(class_name)
            self.centroids[class_name] = np.mean(embeddings, axis=0)

    def predict(self, query_embedding: np.ndarray) -> Dict:
        if not self.centroids:
            raise ValueError("Classifier has not been fitted. Call fit() first.")

        distances = []
        for class_name, centroid in self.centroids.items():
            dist = self.similarity_metric(query_embedding, centroid)
            distances.append((dist, class_name))
        
        # Sort by distance (descending for similarity, ascending for distance)
        distances.sort(key=lambda x: x[0], reverse=True)
        
        confidence, prediction = distances[0]
        
        return {
            "prediction": prediction,
            "confidence": confidence,
        }

    def get_config(self):
        return {"similarity_metric": self.similarity_metric_name}

    def get_name(self):
        return "nearest_centroid"
