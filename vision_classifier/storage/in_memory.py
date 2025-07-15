import pickle
from typing import List, Dict
import numpy as np
from vision_classifier.storage.base import Storage
from datetime import datetime

class InMemoryStorage(Storage):
    def __init__(self):
        self.class_embeddings: Dict[str, List[np.ndarray]] = {}
        self.class_examples: Dict[str, List[str]] = {}
        self.class_images: Dict[str, List[bytes]] = {}

    def add_embedding(self, class_name: str, embedding: np.ndarray, example_path: str, image: bytes = None):
        if class_name not in self.class_embeddings:
            self.class_embeddings[class_name] = []
            self.class_examples[class_name] = []
            self.class_images[class_name] = []
        self.class_embeddings[class_name].append(embedding)
        self.class_examples[class_name].append(example_path)
        self.class_images[class_name].append(image)

    def get_embeddings(self, class_name: str) -> List[np.ndarray]:
        return self.class_embeddings.get(class_name, [])

    def get_all_classes(self) -> List[str]:
        return list(self.class_embeddings.keys())

    def save(self, path: str, encoder: "Encoder", classifier: "Classifier"):
        state = {
            "encoder_metadata": {
                "name": encoder.get_name(),
                "config": encoder.get_config(),
            },
            "classifier_metadata": {
                "name": classifier.get_name(),
                "config": classifier.get_config(),
            },
            "data": {
                "class_embeddings": self.class_embeddings,
                "class_examples": self.class_examples,
                "class_images": self.class_images,
            },
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"System state saved to {path}")

    def load(self, path: str, encoder: "Encoder", classifier: "Classifier"):
        with open(path, "rb") as f:
            state = pickle.load(f)

        if state["encoder_metadata"]["name"] != encoder.get_name():
            raise ValueError(
                f"Encoder mismatch: expected {encoder.get_name()}, "
                f"got {state['encoder_metadata']['name']}"
            )
        if state["classifier_metadata"]["name"] != classifier.get_name():
            raise ValueError(
                f"Classifier mismatch: expected {classifier.get_name()}, "
                f"got {state['classifier_metadata']['name']}"
            )

        self.class_embeddings = state["data"]["class_embeddings"]
        self.class_examples = state["data"]["class_examples"]
        self.class_images = state["data"].get("class_images", {})
        print(f"System state loaded from {path}")
        print(f"Available classes: {list(self.class_embeddings.keys())}")
