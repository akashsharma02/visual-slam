from typing import Union
import numpy as np
import gtsam

class SE3Pose(object):

    """ SE3Pose (Special Euclidean group in 3 dimensions) class represents transformation in 3D space """

    def __init__(self, pose: gtsam.Pose3=gtsam.Pose3()) -> None:
        self.set(pose)
        self.covariance = np.eye(6)

    def set(self, pose: Union[gtsam.Pose3, np.ndarray]=None) -> None:
        """TODO: Docstring for set.

        :function:
            pose: gtsam Pose3 object representing transfomation in 3D space or
                  4x4 numpy ndarray representing transformation in 3D space
        :returns:
            None

        """
        assert pose is not None, "Cannot set 'None' for SE3Pose, expected gtsam Pose3 object"
        if isinstance(pose, gtsam.Pose3):
            self._pose = pose
        elif isinstance(pose, np.ndarray):
            assert pose.shape == (4, 4), "Invalid pose matrix shape, SE3Pose requires 4x4"
            self._pose = gtsam.Pose3(pose)

        self.matrix = self._pose.matrix()
        self.inv_matrix = np.linalg.inv(self._pose.matrix())
        self.rot_matrix = self._pose.rotation().matrix()
        self.trans_matrix = self._pose.translation()



