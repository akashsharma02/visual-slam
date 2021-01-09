import os, sys
import pathlib
from abc import ABC, abstractmethod

class Dataloader(ABC):

    """Abstract base class for data loaders for different datasets
        1. KITTI
        2. TUM
        3. EUROC
    """
    def __init__(self, path: str, cam_param_path: str) -> None:
        self.path = pathlib.Path(path)
        assert self.path.exists()
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
    def getCameraParameters(self) -> dict:
        """ Returns camera parameters as a dictionary
        """
        pass

