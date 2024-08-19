# base_model.py
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        Forward pass logic for the model. Must be overridden by all subclasses.
        """
        raise NotImplementedError("Must define forward() method for subclass.")

    @abstractmethod
    def init_weights(self):
        """
        Initialize weights method. Can be overridden by subclasses if model-specific
        weight initialization is needed.
        """
        pass
