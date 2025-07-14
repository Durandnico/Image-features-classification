from typing import List
from .encoders.base import Encoder
from .classifiers.base import Classifier
from .storage.base import Storage
from .storage.in_memory import InMemoryStorage
from .registry import ENCODERS, CLASSIFIERS
import pickle

class VisionClassifier:
    def __init__(self, encoder: Encoder, classifier: Classifier, storage: Storage):
        self.encoder = encoder
        self.classifier = classifier
        self.storage = storage

    def add_examples(self, class_name: str, image_paths: List[str]):
        for img_path in image_paths:
            try:
                embedding = self.encoder.encode(img_path)
                self.storage.add_embedding(class_name, embedding, img_path)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        print(f"Added {len(image_paths)} examples for class '{class_name}'")

    def train(self):
        self.classifier.fit(self.storage)
        print("Classifier trained.")

    def classify_image(self, image_path: str):
        if not self.storage.get_all_classes():
            raise ValueError("No examples added yet. Add examples and train first.")
        query_embedding = self.encoder.encode(image_path)
        return self.classifier.predict(query_embedding)

    def save(self, path: str = "vision_classifier_state.pkl"):
        self.storage.save(path, self.encoder, self.classifier)

    def load(self, path: str):
        self.storage.load(path, self.encoder, self.classifier)

    @staticmethod
    def load_from_pretrained(path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)

        encoder_name = state["encoder_metadata"]["name"]
        encoder_config = state["encoder_metadata"]["config"]
        
        try:
            if encoder_name == "huggingface":
                from .encoders.hf_encoder import HuggingFaceEncoder
            elif encoder_name == "ollama":
                from .encoders.ollama_encoder import OllamaEncoder
        except ImportError:
            raise ImportError(f"Could not import {encoder_name} encoder. Please install the required dependencies.")

        encoder_class = ENCODERS[encoder_name]
        encoder = encoder_class(**encoder_config)

        classifier_name = state["classifier_metadata"]["name"]
        classifier_config = state["classifier_metadata"]["config"]

        try:
            if classifier_name == "knn":
                from .classifiers.knn import KNNClassifier
            elif classifier_name == "nearest_neighbor":
                from .classifiers.nearest_neighbor import NearestNeighborClassifier
        except ImportError:
            raise ImportError(f"Could not import {classifier_name} classifier. Please install the required dependencies.")

        classifier_class = CLASSIFIERS[classifier_name]
        classifier = classifier_class(**classifier_config)

        storage = InMemoryStorage()
        storage.class_embeddings = state["data"]["class_embeddings"]
        storage.class_examples = state["data"]["class_examples"]

        vision_classifier = VisionClassifier(encoder, classifier, storage)
        vision_classifier.train()
        return vision_classifier
