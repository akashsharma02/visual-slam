from __future__ import annotations
from typing import Optional, List
import numpy as np
import cv2
import gtsam

from vslam.parser import CfgNode
from vslam.types import PinholeCamera
from vslam.feature import FeatureExtractor

class Frame(object):

    """A posed Frame object that holds keypoints, descriptors, triangulated 3D points """

    def __init__(self,
                 image: np.ndarray,
                 timestamp: int,
                 config: CfgNode,
                 camera: PinholeCamera,
                 feature_tracker : FeatureExtractor = None,
                 pose: Optional[gtsam.Pose3] = None) -> None:

        _, _, c = image.shape
        if(c == 3):
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif(c == 4):
            self.image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            msg_no_channels = "Incorrect number of channels to create frame"
            assert c == 1, msg_no_channels
            self.image = image

        self.timestamp = timestamp
        self.config = config

        self.camera = camera

        if pose:
            self.pose = pose
        else:
            self.pose = gtsam.Pose3()

        kps = None
        if feature_tracker is None:
            # TODO: Add custom feature tracker for KLT + ORB descriptor
            self.kps = cv2.goodFeaturesToTrack(self.image,
                                               self.config.num_interest_points,
                                               self.config.quality_level,
                                               self.config.nms_radius)
            kps = _corners_to_keypoints(self.kps)
            orb = cv2.ORB_create(self.config.num_interest_points,
                                 scaleFactor=1.2,
                                 nlevels=3)
            kps, self.des = orb.compute(self.image, kps)
        else:
            # TODO: Add custom feature tracker wrapper
            self.kps = feature_tracker.detectAndCompute(self.image)

        self.kps_un = camera.undistortPoints(self.kps)
        self.kps = np.asarray(kps)
        self.kps_un = np.asarray(_corners_to_keypoints(self.kps_un))

        #TODO: Vector of points in map corresponding to the keypoints
        self.map_points = None

    @property
    def width(self):
        return self.camera.width

    @property
    def height(self):
        return self.camera.height

    def updatePose(self, pose: gtsam.Pose3) -> None:
        if isinstance(pose, np.ndarray):
            self.pose = gtsam.Pose3(pose)
        else:
            self.pose = pose

    def projectPoints(self, points_w: np.ndarray) -> np.ndarray:
        """ Project points in world coordinate frame into the camera projective space

        :function:
            points_w: Nx3 3D points in world coordinate frame
        :returns:
            points: Nx2 2D points in image coordinates (u, v)

        """
        points_c = self.pose.rotation().matrix() @ points_w.T + self.pose.translation()
        return self.camera.project(points_c)

    @staticmethod
    def matchBetween(frame1: Frame, frame2: Frame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute matches between keypoints of input pair of frames

        :Args:
            frame1: first frame
            frame2: second frame
        :returns:
            Tuple containing 2D matched keypoints from database and query frames respectively
        """
        prev_des, curr_des = frame1.des, frame2.des
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(prev_des, curr_des, k=2)

        # Lowe's ratio test
        good, pts1, pts2 = [], [], []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                good.append(m)
                pts2.append(frame2.kps[m.trainIdx].pt)
                pts1.append(frame1.kps[m.queryIdx].pt)

        msg_mismatch = "Number of matched keypoints should be same in previous and current frame"
        assert len(pts1) == len(pts2), msg_mismatch

        return (np.int32(pts1), np.int32(pts2))

    @staticmethod
    def drawFeatures(frame: Frame) -> np.ndarray:
        """
        Draw features on the image

        :function:
            Image: HxW np.ndarray input image
            kps: tracked 2D keypoints
            max_track_length: feature track terminating length

        :returns:
            Image with features drawn

        """
        # image = image.copy()
        image = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
        for kp in frame.kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.drawMarker(image, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=1, markerSize=10)
        return image

    @staticmethod
    def drawMatches(frame: Frame, pts_prev: np.ndarray, pts_curr: np.ndarray) -> np.ndarray:
        """
        Draw Matches on the image

        :Args:
            frame: Frame to draw on
            pts_prev: matched points/corners from previous frame
            pts_curr: matched points/corners from current frame
        :returns:
            modified image with drawing

        """
        image = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
        for kp1, kp2 in zip(pts_prev, pts_curr):
            x1, y1 = int(kp1[0]), int(kp1[1])
            x2, y2 = int(kp2[0]), int(kp2[1])
            cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return image
def _corners_to_keypoints(corners: np.ndarray) -> List:
    if corners is None:
        keypoints = []
    else:
        keypoints = [
            cv2.KeyPoint(corner[0][0], corner[0][1], 1) for corner in corners
        ]
    return keypoints


def _keypoints_to_corners(keypoints: List) -> np.ndarray:
    corners = []
    for kp in keypoints:
        corners.append([kp.pt.x, kp.pt.y])
    return np.asarray(corners)
