from typing import List, Tuple, Optional
import operator
import numpy as np
from .feature import FeatureExtractor
import cv2
import time

class ORBFeatureExtractor(FeatureExtractor):
    """
    Feature Extractor for ORB
    """
    def __init__(self, **kwargs) -> None:
        """Contructor
            Args:
                ORB initializer list specified from config file
        """
        super().__init__()
        self.orb = cv2.ORB_create(**kwargs)
        self.num_interest_points = self.orb.getMaxFeatures()

    def detect(self, img: np.ndarray, mask: np.ndarray = None) -> List:
        """Wrapper class for ORB detector
            Grids the image and run Detection over each
            patch to prevent clustering of ORB features

            Args:
                Image
                Mask
            Returns:
                Length N List of Keypoints 
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        W = 300 # Constant
        (img_h, img_w) = img.shape
        n_cols = np.round(img_w / W).astype(int)
        n_rows = np.round(img_h / W).astype(int)
        cell_width = np.ceil(img_w / n_cols)
        cell_height = np.ceil(img_h / n_rows)
        all_kps = list()

        for i in range(n_rows):
            y_ini = np.round(i * cell_height).astype(int)
            y_max = np.min([y_ini + cell_height, img_h]).astype(int)
            if y_ini >= img_h:
                continue

            for j in range(n_cols):
                x_ini = np.round(j * cell_width).astype(int)
                x_max = np.min([x_ini + cell_width, img_w]).astype(int)
                if x_ini >= img_w:
                    continue

                kps = self.orb.detect(img[y_ini : y_max, x_ini : x_max])
                for kp in kps:
                    kp.pt = ((kp.pt[0] + j * cell_width), (kp.pt[1] + i * cell_height))

                all_kps.extend(kps)

        return self._retainBest(all_kps, self.num_interest_points)

    def compute(self, img: np.ndarray, kp: List) -> Tuple[List, np.ndarray]:
        """Wrapper class for ORB compute
            Args:
                Image

            Returns:
                Length N List of Keypoints 
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.orb.compute(img, kp)

    def _retainBest(self, kps : List, num_interest_points : int) -> List:
        """Keypoint filter to retain best kps based on response
            Args:
                Keypoints
                number of interest points

            Return:
                Keypoints
        """
        if len(kps) < num_interest_points:
            return kps

        if num_interest_points == 0:
            return list()

        kps.sort(key=lambda x : x.response, reverse=True)
        return kps[:num_interest_points]        

    def detectAndCompute(self, img: np.ndarray, mask : np.ndarray = None) -> Tuple[List, np.ndarray]:
        """Wrapper class for ORB detectAndCompute
            Args:
                Image
                Mask

            Returns:
                Tuple of Keypoint and descriptors
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints = self.detect(img, mask)
        print("Key points num: {}".format(len(keypoints)))
        (keypoints, descriptor) = self.compute(img, keypoints)
        return (keypoints, descriptor)
