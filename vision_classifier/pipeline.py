from typing import List
from .encoders.base import Encoder
from .classifiers.base import Classifier
from .storage.base import Storage
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
        with open(path, "wb") as f:
            pickle.dump({
                'storage': self.storage,
                'classifier': self.classifier
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.storage = data['storage']
        self.classifier = data['classifier']
