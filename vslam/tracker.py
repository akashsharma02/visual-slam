import numpy as np
import cv2

class Tracker:

    """
        Implements Tracking functionality
        Estimates camera pose and establishes 3D landmarks for subsequent tracking

    """

    def __init__(self):
        """
        TODO:
        """

    def bootstrap(self, prev_image: cv2.image, curr_image: cv2.image, intrinsic_matrix: np.ndarray=None) -> dict:
        """

        Args:
            intrinsic_matrix: 3x3 Camera intrinsics matrix
            prev_image: rectified first image from monocular camera
            curr_image: rectified second image from monocular camera

        Returns:
            Dictionary containing
                camera pose of second image w.r.t world (1st frame is origin)
                2D keypoints matched in `curr_image`
                3D landmarks after triangulating `curr_image` and `prev_image` matches

        """
        pass

    def processFrame(self, prev_image: cv2.image, curr_image: cv2.image):
        """TODO: Docstring for processFrame.

        :function: TODO
        :returns: TODO

        """
        pass

