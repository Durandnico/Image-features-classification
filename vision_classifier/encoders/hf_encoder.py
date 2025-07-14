import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoVideoProcessor
import numpy as np
from vision_classifier.encoders.base import Encoder
from vision_classifier.registry import register_encoder

def attention_pooling(embeddings):
    """
    embeddings: Tensor of shape [seq_len, hidden_dim]
    returns: Tensor of shape [1, hidden_dim]
    """
    attention_scores = torch.tanh(embeddings)
    attention_weights = F.softmax(attention_scores.mean(dim=1), dim=0)
    pooled = torch.sum(attention_weights.unsqueeze(1) * embeddings, dim=0, keepdim=True)
    return pooled

@register_encoder("huggingface")
class HuggingFaceEncoder(Encoder):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.processor_type = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model {self.model_name} on {self.device}...")
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.processor_type = "image"
        except Exception:
            try:
                self.processor = AutoVideoProcessor.from_pretrained(self.model_name)
                self.processor_type = "video"
            except Exception:
                raise ValueError(f"Could not load processor for model {self.model_name}.")
        print("Model loaded successfully!")

    def encode(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return self._get_embedding(image)

    def _get_embedding(self, image):
        with torch.no_grad():
            if self.processor_type == "image":
                inputs = self.processor(text="", images=image, return_tensors="pt").to(self.device)
                if 'input_ids' in inputs:
                    del inputs['input_ids']
                if 'attention_mask' in inputs:
                    del inputs['attention_mask']
                features = self.model.get_image_features(**inputs)
            elif self.processor_type == "video":
                NUMBER_OF_FRAMES = 4
                pixel_values = self.processor(image, return_tensors="pt").to(self.device)["pixel_values_videos"]
                pixel_values = pixel_values.repeat(1, NUMBER_OF_FRAMES, 1, 1, 1)
                features = self.model.get_vision_features(pixel_values)
            else:
                raise ValueError("Unsupported processor type.")

            if len(features.shape) > 2:
                if features.shape[0] == 1:
                    features = features.squeeze(0)
                else:
                    raise ValueError("Features should be 2D or 3D tensor with batch size of 1.")
            
            if features.shape[0] > 1:
                features = attention_pooling(features)

            embedding = features / features.norm(dim=1, keepdim=True)
            return embedding.cpu().numpy()

    def get_config(self):
        return {"model_name": self.model_name, "device": self.device}

    def get_name(self):
        return "huggingface"
