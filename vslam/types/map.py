import numpy as np
import cv2
import gtsam

from collections import deque

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
        self.config = config
        self.frames = deque(maxlen=self.config["frame_max_length"]
        # TODO: keyframes
        self.points = None

