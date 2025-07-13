from abc import ABC, abstractmethod
import numpy as np

class Encoder(ABC):
    @abstractmethod
    def encode(self, data) -> np.ndarray:
        """Encode data into a numpy array."""
        pass
