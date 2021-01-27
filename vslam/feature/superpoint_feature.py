from typing import List, Tuple, Optional
import numpy as np
import cv2
from .feature import FeatureExtractor
from thirdparty.SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend, PointTracker, VideoStreamer

class SuperPointFeatureExtractor(FeatureExtractor):
    """
    Feature Extractor for SuperPoint
    """
    def __init__(self, **kwargs) -> None:
        """Contructor
            TODO: SuperPoint natively has trackers for points matched across all frames
        """
        super().__init__()
        self.img_h = kwargs['img_h']
        self.img_w = kwargs['img_w']
        
        self.fe = SuperPointFrontend(
                        weights_path=kwargs['weights_path'],
                        nms_dist=kwargs['nms_dist'],
                        conf_thresh=kwargs['conf_thresh'],
                        nn_thresh=kwargs['nn_thresh'],
                        cuda=kwargs['cuda'])

    def detect(self, img: np.ndarray, mask: np.ndarray = None) -> List:
        """Wrapper class for SIFT detector
            Args:
                Image
                Mask

            Returns:
                List of keypoints
        """
        pass

    def compute(self, img: np.ndarray, kp: List) -> Tuple[List, np.ndarray]:
        """Wrapper class for SIFT compute
            Args:
                Image

            Returns:
                Tuple of keypoint and corresponding feature descriptor
        """
        pass

    def _pts_to_keypoints(self, pts : np.ndarray, orig_shape : Tuple, resized_shape : Tuple) -> List:
        """Convert Superpoint generated interest points to cv2 Keypoints
            Args:
                TODO
            
            Returns:
                TODO

            TODO: pts returns a third element, not yet clear what is this element

        """
        pts = pts.astype('float32').T
        # Roughly reproject keypoints to image location prior to resize
        pts[:, 0] = pts[:, 0] / resized_shape[0] * orig_shape[0]
        pts[:, 1] = pts[:, 1] / resized_shape[1] * orig_shape[1]
        kpts = [ cv2.KeyPoint(pt[0], pt[1], 1) for pt in pts]
        return kpts
            
    
    def detectAndCompute(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        """Wrapper class for SuperPoint detectAndCompute
            Args:
                Image

            Returns:
                Tuple of Keypoint and descriptors
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        gray_img = (gray_img.astype('float32') / 255.)
        pts, desc, heatmap = self.fe.run(gray_img)
        kpts = self._pts_to_keypoints(pts, img.shape, (self.img_h, self.img_w))
        return kpts, desc.T
        