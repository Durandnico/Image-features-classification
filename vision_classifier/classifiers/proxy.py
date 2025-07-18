from typing import Dict, Any, List
import numpy as np
from vision_classifier.storage.base import Storage
from vision_classifier.classifiers.base import Classifier


class ProxyClassifier(Classifier):
    def __init__(self, classifiers: List[Classifier]):
        self.classifiers = classifiers

    def fit(self, storage: Storage):
        for classifier in self.classifiers:
            classifier.fit(storage)

    def predict(self, query_embedding: np.ndarray) -> Dict:
        predictions = {}
        for classifier in self.classifiers:
            predictions[classifier.get_name()] = classifier.predict(query_embedding)
        return predictions

    def get_config(self) -> Dict[str, Any]:
        return {
                "proxy": {classifier.get_name(): classifier.get_config() for classifier in self.classifiers}
        }

    def get_name(self) -> str:
        return "proxy"
