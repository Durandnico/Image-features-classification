from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from vision_classifier.storage.base import Storage

class Classifier(ABC):
    @abstractmethod
    def fit(self, storage: Storage):
        pass

    @abstractmethod
    def predict(self, query_embedding: np.ndarray) -> Dict:
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
