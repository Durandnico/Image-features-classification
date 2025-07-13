import ollama
import numpy as np
from .base import Encoder

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
