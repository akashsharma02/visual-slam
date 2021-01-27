from typing import List, Tuple, Optional
import numpy as np
from .feature import FeatureExtractor
import cv2

class SIFTFeatureExtractor(FeatureExtractor):
    """
    Feature Extractor for SIFT
    """
    def __init__(self, **kwargs) -> None:
        """Contructor
            Args:
                SIFT initializer list as specified from config file
        """
        super().__init__()
        self.sift = cv2.SIFT_create(**kwargs)

    def detect(self, img: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Wrapper class for SIFT detector
            Args:
                Image: ndarray
                Mask: ndarray

            Returns:
                List of Keypoints 
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return  self.sift.detect(img, mask=mask)

    def compute(self, img: np.ndarray, kp: List) -> Tuple[List, np.ndarray]:
        """Wrapper class for SIFT compute
            Args:
                Image

            Returns:
                Tuple of keypoint and corresponding feature descriptor
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.sift.compute(img, kp)

    def detectAndCompute(self, img: np.ndarray, mask : np.ndarray = None) -> Tuple[List, np.ndarray]:
        """Wrapper class for SIFT detectAndCompute
            Args:
                Image
                Mask

            Returns:
                Tuple of Keypoint and descriptors
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(img, mask = mask)
