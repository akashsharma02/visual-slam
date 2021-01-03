import pytest
import os, sys
import numpy as np
import cv2

from vslam import dataloaders as loader



class TestKittiDataloader(object):

    """Test dataloaders base class"""

    def test_init(self):
        """ Test KittiDataloader initialization

        Args:
        Returns:

        """
        # kitti = KittiDataloader("~/Documents/datasets/kitti/2011_09_26/", "13")
        pass


class TestTumDataloader(object):

    """Docstring for TestTumDataloader. """


    path = "/home/akashsharma/Documents/datasets/tum/rgbd_dataset_freiburg1_xyz/"
    sequence = "1"
    def test_init(self):
        """ Test TumDataloader initialization

        Args:
        Returns:

        """
        with pytest.raises(AssertionError):
            tum = loader.TumDataloader("/home/akashsharma/Documents/datasets/tum/rgbd_dataset_freiburg1_xyz/", "0")

        with pytest.raises(AssertionError):
            tum = loader.TumDataloader("SomeRandomPath/path", "1")

        tum = loader.TumDataloader(TestTumDataloader.path, TestTumDataloader.sequence)
        assert tum.sequence == 1
        assert len(tum._matches) <= len(tum._readFileList(tum.path / "rgb.txt"))

    def test_len(self):
        """Docstring for test_len.

        :Args:
        :Returns:

        """
        tum = loader.TumDataloader(TestTumDataloader.path, TestTumDataloader.sequence)
        assert len(tum) == len(tum._matches)

    def test_get_item(self):
        """Docstring for test_get_item.

        :Args:
        :Returns:

        """
        tum = loader.TumDataloader(TestTumDataloader.path, TestTumDataloader.sequence)

        for i in [0, len(tum)-1]:
            data_dict = tum[i]
            rgb = data_dict["rgb"]
            depth = data_dict["depth"]
            cam_params = data_dict["cam_params"]

            assert rgb.dtype == np.uint8
            assert rgb.shape == (480, 640, 3)
            assert depth.dtype == np.uint8
            assert depth.shape == (480, 640, 3)
            cv2.imshow("test_get_item", rgb)
            cv2.waitKey(100);
            cv2.imshow("test_get_item", depth)
            cv2.waitKey(100)

            assert cam_params["intrinsic_matrix"].shape == (3, 3)




