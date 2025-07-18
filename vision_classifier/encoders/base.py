from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class Encoder(ABC):
    def __init__(self, device: str = "cpu"):
        self.device = device

    @abstractmethod
    def encode(self, data) -> np.ndarray:
        """Encode data into a numpy array."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def to(self, device: str):
        pass
