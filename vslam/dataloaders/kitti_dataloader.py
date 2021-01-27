import numpy as np
import cv2
import pykitti

from .dataloader import Dataloader

class KittiDataloader(Dataloader):
    """
    Data loader for KITTI dataset
    """
    def __init__(self, path: str, sequence: str) -> None:
        """
        Args:
            path: Path to KITTI Odometry dataset
            sequence: sequence number
        """
        super().__init__(path, path)
        self.sequence = sequence
        self._kitti = pykitti.odometry(path, sequence)
        self.camera_params = self.getCameraParameters()

    def __len__(self) -> int:
        """
        Returns length of sequence
        """
        return len(self._kitti)

    def __getitem__(self, index: int) -> dict:
        """
        Return a dictionary of relevant data
        """
        img = self._kitti.get_cam2(index)
        timestamp = self._kitti.timestamps[index]
        img_np = np.array(img)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        data_dict = {
            "rgb": img_cv2,
            "rgb_timestamp": timestamp,
            "cam_params": self.camera_params
        }
        return data_dict

    def getCameraParameters(self) -> dict:
        """
        Returns:
            dict() containing:
            intrinsic_matrix = 3x3 numpy ndarray of camera intrinsics
            dist_coefficients = 1x5 numpy ndarray of distortion coefficients
        """
        camera = dict()
        camera['width'] = 376
        camera['height'] = 1241
        camera['fx'] = self._kitti.calib.K_cam2[0][0]
        camera['fy'] = self._kitti.calib.K_cam2[1][1]
        camera['cx'] = self._kitti.calib.K_cam2[0][2]
        camera['cy'] = self._kitti.calib.K_cam2[1][2]

        # TODO(Akash): Check for the distortion coefficients
        camera["dist_coefficients"] = np.zeros(5)
        return camera
