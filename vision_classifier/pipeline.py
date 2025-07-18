from typing import List
from vision_classifier.encoders.base import Encoder
from vision_classifier.classifiers.base import Classifier
from vision_classifier.storage.base import Storage
from vision_classifier.storage.in_memory import InMemoryStorage
from vision_classifier.registry import ENCODERS, CLASSIFIERS
import pickle

class VisionClassifier:
    def __init__(self, encoder: Encoder, classifier: Classifier, storage: Storage, store_images: bool = False):
        self.encoder = encoder
        self.classifier = classifier
        self.storage = storage
        self.store_images = store_images

    def add_examples(self, class_name: str, image_paths: List[str]):
        for img_path in image_paths:
            try:
                embedding = self.encoder.encode(img_path)
                image = None
                if self.store_images:
                    with open(img_path, "rb") as f:
                        image = f.read()
                self.storage.add_embedding(class_name, embedding, img_path, image=image)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    def train(self):
        self.classifier.fit(self.storage)
        print("Classifier trained.")

    def set_classifier(self, new_classifier: Classifier):
        self.classifier = new_classifier
        self.train()

    def classify_image(self, image_path: str):
        if not self.storage.get_all_classes():
            raise ValueError("No examples added yet. Add examples and train first.")
        query_embedding = self.encoder.encode(image_path)
        print(f"Query embedding for {image_path}: ")
        print(query_embedding)
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
        storage.class_images = state["data"].get("class_images", {})

        vision_classifier = VisionClassifier(encoder, classifier, storage)
        vision_classifier.train()
        return vision_classifier
