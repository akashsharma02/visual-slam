import numpy as np
import pykitti

class KittiDataLoader(DataLoader):
    """
    Data loader for KITTI dataset
    """
    def __init__(self, data_path: str, sequence: str) -> None:
        """
        Args:
            data_path: Path to KITTI Odometry dataset
            sequence: sequence number
        """
        super().__init__(data_path, data_path)
        self.sequence = sequence
        self._kitti = pykitti.odometry(data_path, sequence)
        self.camera_params = self.getCameraParameters()

    def __len__(self) -> int:
        """
        Returns length of sequence
        """
        return len(self._kitti.poses)

    def __getitem__(self, index: int) -> dict:
        """
        Return a dictionary of relevant data
        """
        image = self._kitti.get_cam2(index)
        data_dict = {
            "image": image,
            "cam_params": self.camera
        }
        return data_dict

    def getCameraParameters(self) -> dict():
        """
        Returns:
            dict() containing:
            intrinsic_matrix = 3x3 numpy ndarray of camera intrinsics
            dist_coefficients = 1x5 numpy ndarray of distortion coefficients
        """
        camera = dict()
        camera.intrinsic_matrix = self._kitti.calib.K_cam2
        # TODO(Akash): Check for the distortion coefficients
        camera.dist_coefficients = np.zeros(5)
        return camera
