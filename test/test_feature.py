import pytest
import numpy as np
import argparse
import tqdm
import cv2
import gtsam

from vslam.parse_config import ConfigParser
from vslam import visualizer as viz
from vslam.types import Camera, PinholeCamera
from vslam.tracker import Tracker

from vslam import dataloaders as loader
from vslam import feature as feat

class TestFeatureExtractor(object):
    """Test Feature Extraction 

    """
    path = "/home/alex/Documents/datasets/rgbd_dataset_freiburg1_xyz/"
    sequence = "1"
    tracker_args = {
        "num_interest_points" : 1000,
        "quality_level" : 0.1,
        "nms_radius" : 2,
        "reproj_threshold" : 3, 
        "confidence" : 0.99
    }
    threshold = 3

    def test_init(self):
        """
        TODO: Obtain all list of feature extractors
        """
        
        orb = feat.ORBFeatureExtractor()
        sift = feat.SIFTFeatureExtractor()
        tum = loader.TumDataloader(TestFeatureExtractor.path, TestFeatureExtractor.sequence)
        cam_params = tum.getCameraParameters()
        camera = PinholeCamera(cam_params['width'], cam_params['height'],
                            cam_params['fx'], cam_params['fy'],
                            cam_params['cx'], cam_params['cy'],
                            cam_params['dist_coefficients'])

        orb_tracker = Tracker(TestFeatureExtractor.tracker_args, camera, orb)
        sift_tracker = Tracker(TestFeatureExtractor.tracker_args, camera, sift)
        native = Tracker(TestFeatureExtractor.tracker_args, camera)
        
        for i in tqdm.tqdm(range(1, len(tum)), desc="TumDataloader"):
            prev_rgb = tum[i - 1]['rgb']
            curr_rgb = tum[i]['rgb']
            
            T_cw_orig, _, _, _, _ = native.bootstrap(prev_rgb, curr_rgb)
            T_cw_sift, _, _, _, _ = sift_tracker.bootstrap(prev_rgb, curr_rgb)
            T_cw_orb, _, _, _, _ = orb_tracker.bootstrap(prev_rgb, curr_rgb)

            err_sift = np.linalg.norm(T_cw_orig.between(T_cw_sift).translation())
            err_orb = np.linalg.norm(T_cw_orig.between(T_cw_orb).translation())
            print("Sift Diff: {}, ORB Diff: {}".format(err_sift, err_orb))
            assert err_sift < TestFeatureExtractor.threshold
            assert err_orb < TestFeatureExtractor.threshold
            
        
    
