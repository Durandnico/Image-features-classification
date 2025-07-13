import pickle
from typing import List, Dict
import numpy as np
from .base import Storage
from datetime import datetime

class InMemoryStorage(Storage):
    def __init__(self):
        self.class_embeddings: Dict[str, List[np.ndarray]] = {}
        self.class_examples: Dict[str, List[str]] = {}

    def add_embedding(self, class_name: str, embedding: np.ndarray, example_path: str):
        if class_name not in self.class_embeddings:
            self.class_embeddings[class_name] = []
            self.class_examples[class_name] = []
        self.class_embeddings[class_name].append(embedding)
        self.class_examples[class_name].append(example_path)

    def get_embeddings(self, class_name: str) -> List[np.ndarray]:
        return self.class_embeddings.get(class_name, [])

    def get_all_classes(self) -> List[str]:
        return list(self.class_embeddings.keys())

    def save(self, path: str = "few_shot_system.pkl"):
        state = {
            "class_embeddings": self.class_embeddings,
            "class_examples": self.class_examples,
            "timestamp": datetime.now().isoformat()
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"System state saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.class_embeddings = state["class_embeddings"]
        self.class_examples = state["class_examples"]
        print(f"System state loaded from {path}")
        print(f"Available classes: {list(self.class_embeddings.keys())}")
