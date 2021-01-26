from multiprocessing import RLock
import numpy as np
import cv2
import gtsam

from .frame import Frame

class Map(object):

    """
    SLAM Map data structure
    """

    def __init__(self, config) -> None:
        """
        self.config = Configuration dictionary subset with "Map" key in yaml configuration file
        self.frames = Deque of given max length storing all incoming frames from dataloader in the map
        self.points = np.ndarray 3xN points of all the points that are in the map at any given time
        """
        # TODO: self._lock = RLock()
        self.config = config
        # dict maintains dictionary order
        self.keyframes = dict(maxlen=self.config.keyframe_max_length)
        self.points = None

    def __len__(self) -> int:
        """
        Returns the number of frames in the map
        """
        return len(self.keyframes)

    def add_keyframe(self, frame: Frame) -> None:
        """
        :Args:
            frame: Frame object to be added
        :returns:
            None

        """
        #TODO: lock to prevent manipulation from different processes
        self.keyframes[frame] = None

    def remove_frame(self, frame: Frame) -> None:
        """
        :Args:
            frame: Frame object to be removed
        :returns:
            None
        """
        try:
            del self.keyframes[frame]
        except:
            raise KeyError(frame)

    def add_points(self, pts3d: np.ndarray) -> None:
        """
        Add points into the map

        :Args:
            pts3d: Nx3 numpy ndarray of 3D triangulated points
        :returns:
            None

        """



