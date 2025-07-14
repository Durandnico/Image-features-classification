import ollama
import numpy as np
from vision_classifier.encoders.base import Encoder
from vision_classifier.registry import register_encoder

@register_encoder("ollama")
class OllamaEncoder(Encoder):
    def __init__(self, model_name='mixtral', host=None):
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def encode(self, image_path: str) -> np.ndarray:
        with open(image_path, "rb") as f:
            image_data = f.read()
        res = self.client.embeddings(
            model=self.model_name,
            prompt=image_data
        )
        return np.array(res["embedding"])

    def get_config(self):
        return {"model_name": self.model_name, "host": self.client.host}

    def get_name(self):
        return "ollama"
