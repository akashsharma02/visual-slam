from typing import List, Tuple, Optional
import numpy as np
from .feature import FeatureExtractor
import cv2

class ORBFeatureExtractor(FeatureExtractor):
    """
    Feature Extractor for ORB
    """
    def __init__(self, **kwargs) -> None:
        """Contructor
            Args: 
        """
        super().__init__()
        self.orb = cv2.ORB_create(**kwargs)

    def detect(self, img: np.ndarray, mask: np.ndarray = None) -> list:
        """Wrapper class for ORB detector
            Args:
                Image
                Mask

            Returns:
                List of keypoints
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.orb.detect(img, mask=mask)

    def compute(self, img: np.ndarray, kp: list) -> Tuple[list, np.ndarray]:
        """Wrapper class for ORB compute
            Args:
                Image

            Returns:
                Tuple of keypoint and corresponding feature descriptor
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.orb.compute(img, kp)

    def detectAndCompute(self, img: np.ndarray, mask : np.ndarray = None) -> Tuple[list, np.ndarray]:
        """Wrapper class for ORB detectAndCompute
            Args:
                Image
                Mask

            Returns:
                Tuple of Keypoint and descriptors
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(img, mask = mask)
                
        