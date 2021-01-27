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
        """
        super().__init__()
        self.orb = cv2.ORB_create(**kwargs)

        self.nlevels = self.orb.getNLevels()
        self.scaleFactor = self.orb.getScaleFactor()
        self.edgeThreshold = self.orb.getEdgeThreshold()
        self.fastThreshold = self.orb.getFastThreshold()
        self.maxFeatures = self.orb.getMaxFeatures()
        self.pathSize = self.orb.getPatchSize()
        self.fast = cv2.FastFeatureDetector_create()

        factor = 1. / self.scaleFactor
        nDesiredFeaturesPerScale = self.maxFeatures * (1 - factor) / (1. - np.power(factor, self.nlevels))
        self.featurePerLevel = np.zeros((self.nlevels))
        sumFeatures = 0
        for i in range(self.nlevels - 1): # baffled at why nlevel - 1
            self.featurePerLevel[i] = np.round(nDesiredFeaturesPerScale).astype(int)
            sumFeatures += self.featurePerLevel[i]
            nDesiredFeaturesPerScale *= factor
        self.featurePerLevel[self.nlevels - 1] = np.max([self.maxFeatures - sumFeatures, 0]).astype(int)

        self.mvScaleFactor = np.arange(self.nlevels)
        self.mvScaleFactor = np.power(self.scaleFactor, self.mvScaleFactor)
        self.PATCH_SIZE = 31

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

    def _computeKeyPoints(self, imgPyrd : list) -> List:
        """Compute raw keypoints

            Args:
                image Pyramid

            Returns:
                list of raw keypoints
        """

        W = 30 # Constant
        total_kps = list()

        for i in range(self.nlevels):
            kps_pyrd = list()
            minBorderX = self.edgeThreshold - 3
            minBorderY = minBorderX
            imgP_h, imgP_w = imgPyrd[i].shape
            maxBorderX = imgP_w - minBorderX
            maxBorderY = imgP_h - minBorderY

            width = maxBorderX - minBorderX
            height = maxBorderY - minBorderY
            nCols = np.round(width / W).astype(int)
            nRows = np.round(height / W).astype(int)
            wCell = np.ceil(width / nCols)
            hCell = np.ceil(height / nRows)

            for j in range(nRows):
                iniY = np.round(minBorderY + j * hCell).astype(int)
                maxY = np.min([iniY + hCell + 6, maxBorderY]).astype(int)

                if (iniY >= maxBorderY - 3):
                    continue

                for k in range(nCols):
                    iniX = np.round(minBorderX + k * wCell).astype(int)
                    maxX = np.min([iniX + wCell + 6, maxBorderX]).astype(int)
                    if (iniX >= maxBorderX - 6):
                        continue

                    # TODO:
                    # 1. Adapt FAST Threshold
                    # Compute keypoint angle
                    kps = self.fast.detect(imgPyrd[i][iniY:maxY, iniX:maxX])
                    for kp in kps:
                        # NOTE: CV2 Keypoint returns reprojected coordinate in Pyramid level 0, while octave
                        # number could be higher
                        kp.pt = ((kp.pt[0] + k * wCell) * np.power(self.scaleFactor, i).astype(float), \
                             (kp.pt[1] + j * hCell) * np.power(self.scaleFactor, i).astype(float))
                        kp.octave = i
                        kp.size = self.PATCH_SIZE*self.mvScaleFactor[i]

                    kps_pyrd.extend(kps)

            total_kps.extend(self._keypointFilter(kps_pyrd, imgPyrd[0].shape, self.featurePerLevel[i]))
        
        return total_kps
            
    def _keypointFilter(self, fastKps: list, imgSize : Tuple, numKps : int) -> list:
        """Filter Keypoints
            Args: 
                Keypoints
                Maximum Number of Keypoint

            Returns:
                Reduced Keypoints
        """
        edge_removed = self._runByImageBorder(fastKps, imgSize)
        retained_best = self._retainBest(edge_removed, int(numKps))
        return retained_best

    def _runByImageBorder(self, kps: list, imgSize : Tuple) -> List:
        """Remove keypoints too close to the image border
            Args:
                Keypoints
                Image Size
                EdgeThreshold

            Return:
                Keypoints not immediatley close to image border
        """

        # TODO: This might be slow
        (h, w) = imgSize
        inner_border = [k for k in kps if ((k.pt[0] > self.edgeThreshold) and (k.pt[0] < w - self.edgeThreshold)) and \
             ((k.pt[1] > self.edgeThreshold) and (k.pt[1] < h - self.edgeThreshold))]
        return inner_border

    def _retainBest(self, kps : list, npoints : int) -> list:
        """Keypoint filter to retain best kps
            Args:
                Keypoints
                Limited Points

            Return:
                Keypoints
        """
        if len(kps) < npoints:
            return kps
        
        if npoints == 0:
            return kps

        kps.sort(key= lambda x : x.response, reverse=True)
        return kps[:npoints]        
    
    def _computePyramid(self, img : np.ndarray) -> List:
        """Compute Pyramid for incoming images
            Args:
                Grey scale image
            
            Returns:
                list of Gaussian Pyramid
                Level 0 is original Resolution
        """
        scale = 1
        (h, w) = img.shape
        imgPyrd = list()
        for i in range(self.nlevels):
            invScale = 1. / scale
            temp = cv2.resize(img, (np.round(w * invScale).astype(int), np.round(h * invScale).astype(int)), interpolation=cv2.INTER_LINEAR)
            # This function auto pads around
            wholeSize = cv2.copyMakeBorder(temp, self.edgeThreshold, self.edgeThreshold,
                self.edgeThreshold, self.edgeThreshold,
                cv2.BORDER_REFLECT_101+cv2.BORDER_ISOLATED)
            imgPyrd.append(wholeSize)
            scale *= self.scaleFactor
        return imgPyrd

    def detectAndCompute(self, img: np.ndarray, mask : np.ndarray = None) -> Tuple[list, np.ndarray]:
        """Wrapper class for ORB detectAndCompute
            Args:
                Image
                Mask

            Returns:
                Tuple of Keypoint and descriptors
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        k = time.perf_counter()
        imgPyrd = self._computePyramid(img)
        q = time.perf_counter() - k
        print(q)
        kps = self._computeKeyPoints(imgPyrd)
        print(time.perf_counter() - q)
        # kps = [k for k in kps if k.octave == 0]
        return self.orb.compute(img, kps)