from typing import List
from vision_classifier.encoders.base import Encoder
from vision_classifier.classifiers.base import Classifier
from vision_classifier.storage.base import Storage
from vision_classifier.storage.in_memory import InMemoryStorage
from vision_classifier.registry import ENCODERS, CLASSIFIERS
import numpy as np
import pickle
import torch

class VisionClassifier:
    def __init__(self, encoder: Encoder, classifier: Classifier, storage: Storage, store_images: bool = False, device: str | None = None):
        self.encoder = encoder
        self.classifier = classifier
        self.storage = storage
        self.store_images = store_images

        new_device = device or self.encoder.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = new_device

        # Set the device for the encoder
        if new_device != self.encoder.device:
            self.encoder.to(new_device)
            self.encoder.device = new_device

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

    def to(self, device: str):
        self.encoder.to(device)

    def _classify_image(self, image_path: str):
        query_embedding = self.encoder.encode(image_path)
        return self.classifier.predict(query_embedding)

    def _classify_array(self, image_array: np.ndarray):
        query_embedding = self.encoder._get_embedding(image_array)
        return self.classifier.predict(query_embedding)

    def classify(self, data: [str, np.ndarray]):
        if not self.storage.get_all_classes():
            raise ValueError("No examples added yet. Add examples and train first.")

        if isinstance(data, str):
            return self._classify_image(data)
        elif isinstance(data, np.ndarray):
            return self._classify_array(data)
        else:
            raise TypeError("Unsupported data type for classification. Please provide a file path (str) or a NumPy array.")

    def save(self, path: str = "vision_classifier_state.pkl"):
        self.storage.save(path, self.encoder, self.classifier)

    def load(self, path: str):
        self.storage.load(path, self.encoder, self.classifier)

    @staticmethod
    def load_from_pretrained(path: str, device: str | None = None):
        with open(path, "rb") as f:
            state = pickle.load(f)

        encoder_name = state["encoder_metadata"]["name"]
        encoder_config = state["encoder_metadata"]["config"]

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        encoder_config["device"] = device
        
        try:
            if encoder_name == "huggingface":
                from .encoders.hf_encoder import HuggingFaceEncoder
            elif encoder_name == "ollama":
                from .encoders.ollama_encoder import OllamaEncoder
        except ImportError:
            raise ImportError(f"Could not import {encoder_name} encoder. Please install the required dependencies.")

        encoder_class = ENCODERS[encoder_name]
        encoder = encoder_class(**encoder_config, device=device)

        classifier_name = state["classifier_metadata"]["name"].split("-")[0]
        classifier_config = state["classifier_metadata"]["config"]

        classifier_class = CLASSIFIERS[classifier_name]
        classifier = classifier_class(**classifier_config)

        storage = InMemoryStorage()
        storage.class_embeddings = state["data"]["class_embeddings"]
        storage.class_examples = state["data"]["class_examples"]
        storage.class_images = state["data"].get("class_images", {})

        vision_classifier = VisionClassifier(encoder, classifier, storage, device=device)
        vision_classifier.train()
        return vision_classifier
