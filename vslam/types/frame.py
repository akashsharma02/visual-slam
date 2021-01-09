import numpy as np
import cv2
import gtsam

from vslam.types.camera import PinholeCamera

class Frame(object):

    """A posed Frame object that holds keypoints, descriptors, triangulated 3D points """

    def __init__(self, id: int, timestamp: int, camera: PinholeCamera, pose: gtsam.Pose3) -> None:

       self.id = id
       self.timestamp = timestamp
       self.camera = camera
       self.pose = pose

       self.kps = None
       self.des = None
       self.pts3d = None
