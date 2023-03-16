import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from torch import Tensor, nn
from typing import Optional
from sklearn.metrics import roc_curve


class Classifier(nn.Module, ABC):
    """
    Wraps a model which produces raw class scores, and provides methods to compute
    class labels and probabilities.
    """

    def __init__(self, model: nn.Module):
        """
        :param model: The wrapped model. Should implement a `forward()` function
        returning (N,C) tensors where C is the number of classes.
        """
        super().__init__()
        self.model = model

        # TODO: Add any additional initializations here, if you need them.
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        # ========================

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (N, D) input tensor, N samples with D features
        :returns: (N, C) i.e. C class scores for each of N samples
        """
        z: Tensor = None

        # TODO: Implement the forward pass, returning raw scores from the wrapped model.
        # ====== YOUR CODE: ======
        z = self.model(x)
        # ========================
        assert z.shape[0] == x.shape[0] and z.ndim == 2, "raw scores should be (N, C)"
        return z

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        :param x: (N, D) input tensor, N samples with D features
        :returns: (N, C) i.e. C probability values between 0 and 1 for each of N
            samples.
        """
        # TODO: Calcualtes class scores for each sample.
        # ====== YOUR CODE: ======
        z = self.forward(x)
        # ========================
        return self.predict_proba_scores(z)

    def predict_proba_scores(self, z: Tensor) -> Tensor:
        """
        :param z: (N, C) scores tensor, e.g. calculated by this model.
        :returns: (N, C) i.e. C probability values between 0 and 1 for each of N
            samples.
        """
        # TODO: Calculate class probabilities for the input.
        # ====== YOUR CODE: ======
        return nn.Softmax(dim=1)(z)
        # ========================

    def classify(self, x: Tensor) -> Tensor:
        """
        :param x: (N, D) input tensor, N samples with D features
        :returns: (N,) tensor of type torch.int containing predicted class labels.
        """
        # Calculate the class probabilities
        y_proba = self.predict_proba(x)
        # Use implementation-specific helper to assign a class based on the
        # probabilities.
        return self._classify(y_proba)

    def classify_scores(self, z: Tensor) -> Tensor:
        """
        :param z: (N, C) scores tensor, e.g. calculated by this model.
        :returns: (N,) tensor of type torch.int containing predicted class labels.
        """
        y_proba = self.predict_proba_scores(z)
        return self._classify(y_proba)

    @abstractmethod
    def _classify(self, y_proba: Tensor) -> Tensor:
        pass


class ArgMaxClassifier(Classifier):
    """
    Multiclass classifier that chooses the maximal-probability class.
    """

    def _classify(self, y_proba: Tensor):
        # TODO:
        #  Classify each sample to one of C classes based on the highest score.
        #  Output should be a (N,) integer tensor.
        # ====== YOUR CODE: ======
        return y_proba.argmax(dim=1)
        # ========================
