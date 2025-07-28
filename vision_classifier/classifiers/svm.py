from typing import Dict
import numpy as np
from vision_classifier.classifiers.base import Classifier
from vision_classifier.storage.base import Storage
from vision_classifier.registry import register_classifier
from sklearn.svm import SVC

@register_classifier("svm")
class SVMClassifier(Classifier):
    def __init__(self, C=1.0, kernel='linear', gamma='scale', probability=True, threshold=0.75):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.threshold = threshold
        self.model = SVC(
            C=self.C, 
            kernel=self.kernel, 
            gamma=self.gamma, 
            probability=self.probability
        )
        self.class_names = []

    def fit(self, storage: Storage):
        self.storage = storage
        embeddings = []
        labels = []
        self.class_names = self.storage.get_all_classes()
        for i, class_name in enumerate(self.class_names):
            class_embeddings = self.storage.get_embeddings(class_name)
            embeddings.extend(class_embeddings)
            labels.extend([i] * len(class_embeddings))
        
        self.model.fit(np.array(embeddings), np.array(labels))

    def predict(self, query_embedding: np.ndarray) -> Dict:
        if self.model is None:
            raise ValueError("Classifier has not been fitted. Call fit() first.")

        prediction_proba = self.model.predict_proba(query_embedding.reshape(1, -1))
        prediction_index = np.argmax(prediction_proba)
        prediction = self.class_names[prediction_index]
        confidence = prediction_proba[0, prediction_index]

        if confidence < self.threshold:
            prediction = "unknown"
        
        return {
            "prediction": prediction,
            "confidence": confidence,
        }

    def get_config(self):
        return {
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "probability": self.probability
        }

    def get_name(self):
        return f"svm-{self.kernel}-C{self.C}-t{self.threshold}"
