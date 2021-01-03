import numpy as np
import glob
import pykitti
import cv2
import os

from vslam.dataloader import Dataloader

class TumDataloader(Dataloader):
    """
    Data loader for TUM dataset
    """
    def __init__(self, path: str, sequence: str) -> None:
        """
        Args:
            path: Path to TUM dataset
            sequence: sequence number (1, 2, or 3) Camera intrinsics are different for each
        """
        super().__init__(path, path)

        self.sequence = int(sequence)
        assert self.sequence in [1, 2, 3], "Invalid sequence for TUM dataset, can be 1, 2, or 3"
        self.camera_params = self.getCameraParameters()

        rgb_files = self._readFileList(self.path / "rgb.txt")
        depth_files = self._readFileList(self.path / "depth.txt")

        self._matches = self._associate(rgb_files, depth_files, 0.0, 0.02)


    def __len__(self) -> int:
        """
        Returns length of sequence
        """
        return len(self._matches)


    def __getitem__(self, index: int) -> dict:
        """
        Return a dictionary of relevant data
        """
        (rgb_stamp, rgb_data), (depth_stamp, depth_data) = self._matches[index]
        rgb = cv2.imread(str(self.path / rgb_data))
        depth = cv2.imread(str(self.path / depth_data))
        data_dict = {
            "rgb": rgb,
            "depth": depth,
            "rgb_stamp": rgb_stamp,
            "depth_stamp": depth_stamp,
            "cam_params": self.camera_params
        }
        return data_dict


    def getCameraParameters(self) -> np.ndarray:
        """
        Returns a 3 X 3 array of the intrinsics
        Returns:
            A ndarray of the rgb-left camera intrinsics
        """
        camera = dict()
        if self.sequence == 1:
            camera['intrinsic_matrix'] = np.array([[517.3, 0    , 318.6],
                                                   [    0, 516.5, 255.3],
                                                   [    0,     0,     1]])
            camera['dist_coefficients'] = np.array([0.2624,	-0.9531, -0.0054, 0.0026, 1.1633])
        elif self.sequence == 2:
            camera['intrinsic_matrix'] = np.array([[520.9,     0, 325.1],
                                                   [    0, 521.0, 249.7],
                                                   [    0,     0,     1]])
            camera['dist_coefficients'] = np.array([0.2312,	-0.7849, -0.0033, -0.0001, 0.9172])
        elif self.sequence == 3:
            camera['intrinsic_matrix'] = np.array([[535.4,     0, 320.1],
                                                   [    0, 539.2, 247.6],
                                                   [    0,     0,     1]])
            camera['dist_coefficients'] = np.zeros(5)
        else:
            assert False, "Invalid sequence number"
        return camera


    def _readFileList(self, filename: str) -> dict:
        """
        Reads a trajectory from a text file.

        File format:
        The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
        and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.
        Args:
            filename
        Output:
            dict: dictionary of (stamp,data) tuples

        """
        print(os.getcwd())
        file = open(filename)
        data = file.read()
        lines = data.replace(","," ").replace("\t"," ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
        list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
        return dict(list)


    def _associate(self, first_list: dict, second_list: dict, offset: float, max_difference: float) -> list:
        """
        Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
        to find the closest match for every input tuple.

        Args:
            first_list:     first dictionary of (stamp,data) tuples
            second_list:    second dictionary of (stamp,data) tuples
            offset:         time offset between both dictionaries (e.g., to model the delay between the sensors)
            max_difference: search radius for candidate generation

        Returns:
            matches:    list of matched tuples ((stamp1,data1),(stamp2,data2))

        """
        first_keys = list(first_list.keys())
        second_keys = list(second_list.keys())

        potential_matches = [(abs(a - (b + offset)), a, b)
                             for a in first_keys
                             for b in second_keys
                             if abs(a - (b + offset)) < max_difference]
        potential_matches.sort()
        matches = []
        for diff, a, b in potential_matches:
            if a in first_keys and b in second_keys:
                first_keys.remove(a)
                second_keys.remove(b)
                matches.append(((a, first_list[a][0]), (b, second_list[b][0])))

        matches.sort()
        return matches


