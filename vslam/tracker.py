from typing import List, Set, Dict, Tuple, Optional
from enum import Enum
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import gtsam

from vslam.camera import PinholeCamera

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

    def __init__(self, config: Dict, camera: PinholeCamera = None) -> None:
        self.config = config
        self.camera = camera
        self.state  = self.State.NOT_INITIALIZED
        self.prev_image = None

    def track(self, curr_image: np.ndarray) -> Dict:
        """

        Main function in Tracker for
        1. Initialization
        2. Tracking by projection
        2. Tracking with previous frame
        3. Tracking with map points

        :function:
            curr_image: rectified image from monocular camera
        :returns:
            TODO:

        """
        if self.state == self.State.NOT_INITIALIZED:
            self.prev_image = curr_image
            self.state = self.State.INITIALIZING
            return
        elif self.state == self.State.INITIALIZING:
            msg_no_prev_image = "Previous image is 'None', cannot initialize Tracker"
            assert self.prev_image != None, msg_no_prev_image
            #TODO: Return success from bootstrap and retry
            self.bootstrap(self.prev_image, curr_image)
            self.state = self.State.RUNNING
        elif self.state == self.State.RUNNING:
            pass


    def bootstrap(self,
                  prev_image: np.ndarray,
                  curr_image: np.ndarray) -> Dict:
        """

        Args:
            prev_image: rectified first image from monocular camera
            curr_image: rectified second image from monocular camera

        Returns:
            Dictionary containing
                camera pose of second image w.r.t world (1st frame is origin)
                2D keypoints matched in `curr_image`
                3D landmarks after triangulating `curr_image` and `prev_image` matches

        """
        # 1. Estimate 2D correspondences between input image pair
        start = time.time()
        prev_matched_kps, curr_matched_kps = self.matchImagePair(
            prev_image, curr_image)
        print(f"{(time.time() - start)*1000} ms for feature matching"
              )  # About 20 ms

        # 2. Estimate Pose using essential matrix (5 point algorithm)
        start = time.time()
        R, t, inlier_kps1, inlier_kps2 = self.estimatePose(prev_matched_kps, curr_matched_kps)
        print(
            f"{(time.time() - start)*1000} ms for fundamental matrix computation"
        )
        T_cw = np.eye(4)
        T_cw[:3, :3], T_cw[:3, 3] = R, t.T
        T_cw = gtsam.Pose3(T_cw)

        #TODO: Enable visualization in debug mode only
        # image_matches = curr_image.copy()
        # print(inlier_kps1.shape)
        # for kp1, kp2 in zip(inlier_kps1, inlier_kps2):
        #     x1, y1 = int(kp1[0]), int(kp1[1])
        #     x2, y2 = int(kp2[0]), int(kp2[1])
        #     cv2.arrowedLine(image_matches, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # plt.imshow(cv2.cvtColor(image_matches, cv2.COLOR_BGR2RGB))
        # plt.show()

        # 3. Triangulate matched keypoints
        M1 = self.camera.intrinsic_matrix @ np.eye(4)[:3, :]
        M2 = self.camera.intrinsic_matrix @ T_cw.matrix()[:3, :]

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
        # TODO: Clean up return by creating map datastructure
        return T_cw, inlier_kps1, inlier_kps2, pts3d, reproj_errors

    def estimatePose(self, kps1: np.ndarray,
            kps2: np.ndarray) -> Tuple[SE3Pose, np.ndarray, np.ndarray]:
        """

        Estimate pose between two frames given the interest points in either of the frames

        Args:
            kps1: Keypoints in the first frame
            kps2: Keypoints in the second frame
        Returns:
            Tuple containing R, t
        """
        F, inlier_mask = _computeFundamentalMatrix(kps1, kps2)

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
        _, R, t, mask = cv2.recoverPose(E, inlier_kps1, inlier_kps2,
                                        self.camera.intrinsic_matrix)
        return R, t, inlier_kps1, inlier_kps2

    def processFrame(self, prev_image: np.ndarray,
                     curr_image: np.ndarray) -> Dict:
        """TODO: Docstring for processFrame.

        Args:
            prev_image: rectified previous image in the sequence from monocular camera
            curr_image: rectified current image in the sequence from monocular camera
        Returns:
            Dictionary containing
                relative camera pose of `curr_image` w.r.t `prev_image`


        """
        pass

    def matchImagePair(
            self, prev_image: np.ndarray,
            curr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """

        Matches database image == `prev_image` with query image == `curr_image` using KLT features and ORB descriptors

        #TODO:
            1. Divide image into subpatches and extract features individually
            2. Potential enhancement using superpoint features
            3. Use different matcher between features in image
        #NOTE:
            1. Corner detection always gives interest points on the edges of the objects obviously, so using object detector
            on it may not give trackable features

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
                                               self.config["num_interest_points"],
                                               self.config["quality_level"],
                                               self.config["nms_radius"])
        #TODO Visualize prev_keypoints for logger in verbose debug mode
        # prev_image_corners = prev_image.copy()
        # cv2.waitKey(0)
        # for corner in prev_corners:
        #     x, y = corner.ravel()
        #     cv2.drawMarker(prev_image_corners, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=1, markerSize=10)
        # cv2.imshow("Corners debug", prev_image_corners)
        # cv2.waitKey(0)

        curr_corners = cv2.goodFeaturesToTrack(curr_image_gray,
                                               self.config["num_interest_points"],
                                               self.config["quality_level"],
                                               self.config["nms_radius"])

        prev_keypoints = _corners_to_keypoints(prev_corners)
        curr_keypoints = _corners_to_keypoints(curr_corners)

        # 2. Compute descriptors for detected features
        orb = cv2.ORB_create(self.config["num_interest_points"],
                             scaleFactor=1.2,
                             nlevels=3)
        prev_keypoints, prev_desc = orb.compute(prev_image_gray,
                                                prev_keypoints)
        curr_keypoints, curr_desc = orb.compute(curr_image_gray,
                                                curr_keypoints)

        # 3. Compute matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(prev_desc, curr_desc, k=2)
        # matches = sorted(matches, key=lambda x: x.distance)

        # Lowe's ratio test
        good, pts1, pts2 = [], [], []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                good.append(m)
                pts2.append(curr_keypoints[m.trainIdx].pt)
                pts1.append(prev_keypoints[m.queryIdx].pt)

        msg_mismatch = "Number of matched keypoints should be same in previous and current frame"
        assert len(pts1) == len(pts2), msg_mismatch

        # for kp1, kp2 in zip(pts1, pts2):
        #     x1, y1 = int(kp1[0]), int(kp1[1])
        #     x2, y2 = int(kp2[0]), int(kp2[1])
        #     image_matches = cv2.arrowedLine(curr_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # img_matches = cv2.drawMatches(prev_image, prev_keypoints, curr_image,
        #                               curr_keypoints, matches, None,
        #                               matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        #                               flags=0)
        # cv2.imshow("matched_keypoints", image_matches)
        # cv2.waitKey(0)
        # cv2.imshow("Matches", img_matches)
        # cv2.waitKey(0)

        return (np.int32(pts1), np.int32(pts2))


def _corners_to_keypoints(corners: np.ndarray) -> List:
    if corners is None:
        keypoints = []
    else:
        keypoints = [
            cv2.KeyPoint(corner[0][0], corner[0][1], 1) for corner in corners
        ]
    return keypoints


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
    #TODO: Use parameters from config file for the options
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
