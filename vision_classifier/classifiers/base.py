from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from ..storage.base import Storage

class Classifier(ABC):
    @abstractmethod
    def fit(self, storage: Storage):
        pass

    @abstractmethod
    def predict(self, query_embedding: np.ndarray) -> Dict:
        pass
