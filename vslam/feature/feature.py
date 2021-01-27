from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np

class FeatureExtractor(ABC):

    """Abstact Base class for feature extractors
    """
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def detect(self, img: np.ndarray) -> List:
        """
        Args:
            Image in the form of ndarray
        Returns:
            List of Keypoints 
        """
        pass

    @abstractmethod
    def compute(self, img: np.ndarray, feat: np.ndarray) -> Tuple[List, np.ndarray]:
        """Compute Descriptors with given interest points

        Args:
            Image
            Interest Points

        Returns:
            Tuple of interest points and corresponing descriptors
            Tuple[Length N list, N x d np.ndarray]
        """
        pass

    @abstractmethod
    def detectAndCompute(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect Interest Points and compute corresponding feature descriptors

        Args:
            Image

        Returns:
            Tuple of interest points and corresponing descriptors
            Tuple[Length N list, N x d np.ndarray]
        """
        pass