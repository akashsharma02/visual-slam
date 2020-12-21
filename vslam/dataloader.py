import os, sys
from abc import ABC

class DataLoader(ABC):

    """Abstract base class for data loaders for different datasets
        1. KITTI
        2. TUM
        3. EUROC
    """
    def __init__(self, data_path: str, cam_param_path: str) -> None:
        self.data_path = data_path
        self.cam_param_path = cam_param_path

        self.files = []

    @abstractmethod
    def __len__(self) -> int:
        """Returns length of the data stream

        Returns:
            length
        """
        return len(self.files)

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """ Returns dictionary containing data at index

        Args:
            index
        Returns:
            dictionary at index
        """
        return self.files[index]

    @abstractmethod
    def getCameraParameters(self):
        """ Returns camera parameters as a dictionary
        """
        pass

