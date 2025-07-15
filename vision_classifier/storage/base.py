from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

class Storage(ABC):
    @abstractmethod
    def add_embedding(self, class_name: str, embedding: np.ndarray, example_path: str, image: bytes = None):
        pass

    @abstractmethod
    def get_embeddings(self, class_name: str) -> List[np.ndarray]:
        pass

    @abstractmethod
    def get_all_classes(self) -> List[str]:
        pass

    @abstractmethod
    def save(self, path: str, encoder: "Encoder", classifier: "Classifier"):
        pass

    @abstractmethod
    def load(self, path: str, encoder: "Encoder", classifier: "Classifier"):
        pass
