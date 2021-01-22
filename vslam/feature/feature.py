from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np

class FeatureExtractor(ABC):

    """Abstact Base class for feature extractors
    """
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def detect(self, img: np.ndarray) -> list:
        """

        Args:
            Image in the form of ndarray
        Returns:
            Interest points 
        """
        pass

    @abstractmethod
    def compute(self, img: np.ndarray, feat: np.ndarray) -> Tuple[list, np.ndarray]:
        """Compute Descriptors with given interest points

        Args:
            Image
            Interest Points

        Returns:
            Tuple of interest points and descriptors
        """
        pass

    @abstractmethod
    def detectAndCompute(self, img: np.ndarray) -> Tuple[list, np.ndarray]:
        """Detect Interest Points and compute corresponding feature descriptors

        Args:
            Image

        Returns:
            Tuple of Keypoint and Corresponing Descriptors
        """
        pass