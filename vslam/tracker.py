from __future__ import annotations
from typing import Dict, Tuple, Optional
from enum import Enum
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import gtsam

<<<<<<< HEAD
from vslam.types.camera import PinholeCamera
from vslam.feature.feature import FeatureExtractor
=======
from vslam.types import PinholeCamera, Frame, Map
from vslam.parser import CfgNode

>>>>>>> tracker

class Tracker:
    """
        Implements Tracking functionality
        Estimates camera pose and establishes 3D landmarks for subsequent tracking

    """
    class State(Enum):
        NOT_INITIALIZED = 0
        INITIALIZING = 1
        RUNNING = 2
        LOST = 3

<<<<<<< HEAD
    def __init__(self, config: Dict, camera: PinholeCamera = None, feature: FeatureExtractor = None) -> None:
=======
    def __init__(self, config: CfgNode, map_config: CfgNode, camera: PinholeCamera = None) -> None:
>>>>>>> tracker
        self.config = config
        self.map_config = map_config
        self.camera = camera
<<<<<<< HEAD
        self.state  = self.State.NOT_INITIALIZED
        self.prev_image = None
        self.feature = feature
=======
        self.state = self.State.NOT_INITIALIZED
        self.initial_frame = None
>>>>>>> tracker

    def track(self, curr_frame: Frame, slam_map: Optional[Map]=None) -> None:
        """

        Main function in Tracker for
        1. Initialization
        2. Tracking by projection
        2. Tracking with previous frame
        3. Tracking with map points

        :function:
            curr_image: rectified image from monocular camera
        :returns:
            None

        """
        if self.state == self.State.NOT_INITIALIZED:
            # If there are enough keypoints in the frame
            if curr_frame.kps.shape[0] > self.config.min_init_interest_points:
                self.initial_frame = curr_frame
            self.state = self.State.INITIALIZING
            return

        elif self.state == self.State.INITIALIZING:
            msg_no_initial_frame = "Previous image is 'None', cannot initialize Tracker"
            assert self.initial_frame != None, msg_no_initial_frame

            print(curr_frame.kps.shape[0])
            # If the number of keypoints in the frame are too few
            if curr_frame.kps.shape[0] < self.config.min_init_interest_points:
                return

            print("Entering bootstrap")
            # TODO: Return success from bootstrap and retry
            T_cprev_ccurr, pts3d, reproj_errors = self.bootstrap(
                self.initial_frame, curr_frame)
            print(T_cprev_ccurr, reproj_errors)
            # If the triangulation of the features is not good
            if reproj_errors[0] > self.config.max_init_reproj_error or \
               reproj_errors[1] > self.config.max_init_reproj_error:
                return

            if slam_map is None:
                slam_map = Map(self.map_config)

            slam_map.add_keyframe(self.initial_frame)
            slam_map.add_keyframe(curr_frame)
            self.state = self.State.RUNNING

        elif self.state == self.State.RUNNING:
            pass

    def bootstrap(self,
                  prev_frame: Frame,
                  curr_frame: Frame) -> Tuple[gtsam.Pose3, np.ndarray, np.ndarray]:
        """

        Args:
            prev_frame: the kps of frame are updated
            curr_frame: the kps of frame are updated

        Returns:
            Dictionary containing
                camera pose of second image w.r.t world (1st frame is origin)
                Nx3 3D landmarks after triangulating `curr_image` and `prev_image` matches
                N reprojection errors for the trinagulated image

        """
        # 1. Estimate 2D correspondences between input image pair
        start = time.time()
        prev_matched_kps, curr_matched_kps = Frame.matchBetween(prev_frame, curr_frame)
        # About 20 ms
        print(f"{(time.time() - start)*1000} ms for feature matching")
        prev_image = Frame.drawFeatures(prev_frame)
        curr_image = Frame.drawFeatures(curr_frame)
        cv2.imshow("Previous image", prev_image)
        cv2.imshow("Current image", curr_image)

        # 2. Estimate Pose using essential matrix (5 point algorithm)
        start = time.time()
        T_cw, inlier_kps1, inlier_kps2 = self.estimatePose(
            prev_matched_kps, curr_matched_kps)
        print(f"{(time.time() - start)*1000} ms for fundamental matrix computation")

        matches_image = Frame.drawMatches(curr_frame, inlier_kps1, inlier_kps2)
        cv2.imshow("Matches between frames", matches_image)
        cv2.waitKey(0)

        # 3. Triangulate matched keypoints
        prev_frame.updatePose(np.eye(4))
        curr_frame.updatePose(T_cw)

        M1 = prev_frame.camera.intrinsic_matrix @ prev_frame.pose.matrix()[:3, :]
        M2 = curr_frame.camera.intrinsic_matrix @ curr_frame.pose.matrix()[:3, :]

        pts3d = _triangulatePoints(inlier_kps1, inlier_kps2, M1, M2)

        # 4. Calculate metrics (reprojection error)
        pts3d_proj_homo1 = pts3d @ M1.T
        pts3d_proj_homo2 = pts3d @ M2.T
        pts3d_proj_homo1 /= pts3d_proj_homo1[:, 2][:, None]
        pts3d_proj_homo2 /= pts3d_proj_homo2[:, 2][:, None]

        error1 = inlier_kps1 - pts3d_proj_homo1[:, :2]
        error1 = np.sum([np.dot(err1, err1) for err1 in error1])
        error1 = np.sqrt(error1) / inlier_kps1.shape[0]

        error2 = inlier_kps2 - pts3d_proj_homo2[:, :2]
        error2 = np.sum([np.dot(err2, err2) for err2 in error2])
        error2 = np.sqrt(error2) / inlier_kps2.shape[0]

        reproj_errors = [error1, error2]

        return T_cw, pts3d, reproj_errors

    def estimatePose(self, kps1: np.ndarray,
                     kps2: np.ndarray) -> Tuple[gtsam.Pose3, np.ndarray, np.ndarray]:
        """

        Estimate pose between two frames given the interest points in either of the frames

        Args:
            kps1: Keypoints in the first frame
            kps2: Keypoints in the second frame
        Returns:
            Tuple containing
                estimated relative pose
                Nx2 inliers from kps1
                Nx2 inliers from kps2
        """
        _, inlier_mask = _computeFundamentalMatrix(kps1, kps2)

        # Use fundamental matrix estimation to remove outliers
        inlier_mask = np.squeeze(inlier_mask).astype(bool)
        inlier_kps1 = kps1[inlier_mask, :]
        inlier_kps2 = kps2[inlier_mask, :]

        E, inlier_mask = cv2.findEssentialMat(
            inlier_kps1,
            inlier_kps2,
            self.camera.intrinsic_matrix,
            method=cv2.RANSAC,
            prob=self.config["confidence"],
            threshold=self.config["reproj_threshold"])
        inlier_mask = np.squeeze(inlier_mask).astype(bool)
        inlier_kps1 = inlier_kps1[inlier_mask, :]
        inlier_kps2 = inlier_kps2[inlier_mask, :]
        _, R, t, _ = cv2.recoverPose(E, inlier_kps1, inlier_kps2,
                                     self.camera.intrinsic_matrix)
        return gtsam.Pose3(gtsam.Rot3(R), t), inlier_kps1, inlier_kps2

    def processFrame(self, prev_image: np.ndarray,
                     curr_image: np.ndarray) -> Dict:
        """

        Args:
            prev_image: rectified previous image in the sequence from monocular camera
            curr_image: rectified current image in the sequence from monocular camera
        Returns:
            Dictionary containing
                relative camera pose of `curr_image` w.r.t `prev_image`


        """
        pass

def _computeFundamentalMatrix(
        kps_ref_2d: np.ndarray,
        kps_curr_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes fundamental matrix for an image pair given keypoints in each image

    Args:
        kps_ref_2d:  2D keypoints (image coords) in first frame
        kps_curr_2d: 2D keypoints (image coords) in second frame
    Returns:
        F: fundamental matrix 3x3
        mask: optional mask to filter outliers

    """
    # TODO: Use parameters from config file for the options
    F, mask = cv2.findFundamentalMat(kps_ref_2d, kps_curr_2d, cv2.FM_RANSAC)
    if F is None or F.shape == (1, 1):
        raise Exception('No Fundamental Matrix found')
    elif (F.shape[0] > 3):
        F = F[0:3, 0:3]
    return np.matrix(F), mask


def _triangulatePoints(kps1: np.ndarray, kps2: np.ndarray, M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """

    Tringulate points given 2D correspondences in a pair of images, and projection matrices

    Args:
        kps1: Nx2 2D keypoints (image coords) in first image
        kps2: Nx2 2D keypoints (image coords) in second image
        M1: 3x4 projection matrix of first image
        M2: 3x4 projection matrix of second image
    Returns:
        Nx4 3D triangulated points in homogeneous coordinates
    """
    kps1_homo = np.ones((3, kps1.shape[0]))
    kps2_homo = np.ones((3, kps2.shape[0]))
    kps1_homo[0, :], kps1_homo[1, :] = kps1[:, 0].copy(), kps1[:, 1].copy()
    kps2_homo[0, :], kps2_homo[1, :] = kps2[:, 0].copy(), kps2[:, 1].copy()

    pts3d = cv2.triangulatePoints(M1, M2, kps1_homo[:2], kps2_homo[:2])
    pts3d /= pts3d[3, :]
    pts3d = pts3d.T
    return pts3d
