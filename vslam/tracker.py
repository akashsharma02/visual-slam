import numpy as np
import cv2
import matplotlib.pyplot as plt

class Tracker:

    """
        Implements Tracking functionality
        Estimates camera pose and establishes 3D landmarks for subsequent tracking

    """

    def __init__(self, config):
        """
        TODO:
        """
        self.config = config

    def bootstrap(self, prev_image: np.ndarray, curr_image: np.ndarray, intrinsic_matrix: np.ndarray=None) -> dict:
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
        # 1. Estimate 2D correspondences between input image pair
        # self.matchImagePair(prev_image, curr_image)
        pass

    def processFrame(self, prev_image: np.ndarray, curr_image: np.ndarray) -> dict:
        """TODO: Docstring for processFrame.

        Args:
            prev_image: rectified previous image in the sequence from monocular camera
            curr_image: rectified current image in the sequence from monocular camera
        Returns:
            Dictionary containing
                relative camera pose of `curr_image` w.r.t `prev_image`


        """
        pass

    def matchImagePair(self, prev_image: np.ndarray, curr_image: np.ndarray) -> tuple:
        """

        Matches database image == `prev_image` with query image == `curr_image` using KLT features and ORB descriptors

        #TODO:
            1. Divide image into subpatches and extract features individually
            2. Potential enhancement using superpoint features

        Args:
            prev_image: rectified previous image in the sequence from monocular camera
            curr_image: rectified current image in the sequence from monocular camera
        Returns:
            Tuple containing 2D matched keypoints from database and query image respectively

        """
        # 1. Detect keypoints in image 1 (prev) and 2 (curr)
        prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        curr_image_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

        prev_corners = cv2.goodFeaturesToTrack(prev_image_gray,
                                               self.config["max_corners"],
                                               self.config["quality_level"],
                                               self.config["nms_radius"])
        #TODO Visualize prev_keypoints for logger in verbose debug mode
        prev_image_corners = prev_image.copy()
        cv2.waitKey(0)
        for corner in prev_corners:
            x, y = corner.ravel()
            cv2.drawMarker(prev_image_corners, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=1, markerSize=10)
        cv2.imshow("Corners debug", prev_image_corners)
        cv2.waitKey(0)

        curr_corners = cv2.goodFeaturesToTrack(curr_image_gray,
                                               self.config["max_corners"],
                                               self.config["quality_level"],
                                               self.config["nms_radius"])
        prev_keypoints = _corners_to_keypoints(prev_corners)
        curr_keypoints = _corners_to_keypoints(curr_corners)
        # 2. Compute descriptors for detected features
        orb = cv2.ORB_create(self.config["max_corners"], scaleFactor=1.2, nlevels=3)
        prev_keypoints, prev_desc = orb.compute(prev_image_gray, prev_keypoints)
        curr_keypoints, curr_desc = orb.compute(curr_image_gray, curr_keypoints)

        # 3. Compute matches
        bf = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING2, crossCheck=True)
        matches = bf.match(prev_desc, curr_desc)

        # img_matches = cv2.drawMatches(prev_image, prev_keypoints, curr_image,
        #                               curr_keypoints, matches, None,
        #                               matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        #                               flags=0)

        # cv2.imshow("Matches", img_matches)
        # cv2.waitKey(0)


def _corners_to_keypoints(corners: np.ndarray) -> list:
    if corners is None:
        keypoints = []
    else:
        keypoints = [cv2.KeyPoint(corner[0][0], corner[0][1], 1) for corner in corners]
    return keypoints








