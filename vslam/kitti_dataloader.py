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
        self.kitti = pykitti.odometry(data_path, sequence)
        self.camera_params = self.getCameraParameters()

    def __len__(self) -> int:
        """
        Returns length of sequence
        """
        return len(self.kitti.poses)

    def __getitem__(self, index: int) -> dict:
        """
        Return a dictionary of relevant data
        """
        image = self.kitt.get_cam2(index)
        data_dict = {
            "image": image,
            "cam_params": self.camera_params
        }
        return data_dict

    def getCameraParameters(self) -> np.ndarray:
        """
        Returns a 3 X 3 array of the intrinsics
        Returns:
            A ndarray of the rgb-left camera intrinsics
        """
        return self.kitti.calib.K_cam2
