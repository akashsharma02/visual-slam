import numpy as np
import cv2


class Camera:
    """Camera model with intrinsic_matrixs and distortion coefficients"""
    def __init__(self,
                 width: int,
                 height: int,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 dist_coefficients: np.ndarray = None) -> None:
        """

        """
        self.width = width
        self.height = height

        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        if dist_coefficients is not None:
            msg_dist_shape = "Invalid distortion coefficients shape to initialize camera, should be 1x5 array"
            assert dist_coefficients.shape == (
                1, 5) or dist_coefficients.shape == (5,), msg_dist_shape
            self.dist_coefficients = dist_coefficients
        else:
            self.dist_coefficients = np.zeros(5)

        self.k1 = self.dist_coefficients[0]
        self.k2 = self.dist_coefficients[1]
        self.p1 = self.dist_coefficients[2]
        self.p2 = self.dist_coefficients[3]
        self.k3 = self.dist_coefficients[4]

        self.distorted = not self.dist_coefficients.any()


class PinholeCamera(Camera):
    """Pinhole Camera model with projection and unprojection functions"""
    def __init__(self,
                 width: int,
                 height: int,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 dist_coefficients: np.ndarray = None) -> None:
        """TODO: to be defined.

        :Docstring for PinholeCamera.: TODO

        """
        super().__init__(width, height, fx, fy, cx, cy, dist_coefficients)
        self.intrinsic_matrix = np.array([[self.fx, 0, self.cx],
                                   [0, self.fy, self.cy],
                                   [0,       0,       1]])
        self.intrinsic_matrix_inv = np.linalg.inv(self.intrinsic_matrix)

    def project(self, pts3d: np.ndarray) -> np.ndarray:
        """ Project 3D points in numpy array into 2D image coordinate with depths

        Args:
            pts3d 3xN numpy array points in 3D space w.r.t camera frame
        Returns:
            pts2d: 2xN numpy array points in 2D image space coordinates
            depths: 1xN numpy array depths at corresponding coordinates
        """
        pts2d = self.intrinsic_matrix @ pts3d
        depths = pts2d[2, :]
        pts2d /= depths
        return pts2d, depths

    def unproject(self, pts2d: np.ndarray) -> np.ndarray:
        """ Project 2D points on image plane into 3D normalized points in world w.r.t camera frame

        Args:
            pts2d: 2xN 2D points in image coordinate space
        Returns:
            pts3d: 3xN 3D normalized points in world coordinate w.r.t camera frame

        """
        pts2d_homo = np.vstack(pts2d, np.ones(pts2d.shape[1]))
        pts3d = self.intrinsic_matrix_inv @ pts2d_homo
        return pts3d
