import numpy as np
import argparse
import tqdm
import cv2
import gtsam

from vslam.parse_config import ConfigParser
from vslam import visualizer as viz
from vslam import dataloaders
from vslam.types import Camera, PinholeCamera
from vslam.tracker import Tracker
from vslam import feature

def main(config):
    """TODO: Docstring for main.

    :function: TODO
    :returns: TODO

    """
    #TODO: Logger
    dataloader = config.init_obj('dataset', dataloaders)
    cam_params = dataloader.getCameraParameters()

    camera = PinholeCamera(cam_params['width'], cam_params['height'],
                           cam_params['fx'], cam_params['fy'],
                           cam_params['cx'], cam_params['cy'],
                           cam_params['dist_coefficients'])

    feat_extractor = config.init_obj('feature', feature)
    tracker = Tracker(config["tracker"]["args"], camera, feat_extractor)

    for i in tqdm.tqdm(range(5, 15), desc=config['dataset']['type']):
        # data = dataloader[i]
        # cv2.imshow("rgb image", data["rgb"])
        # cv2.waitKey(1)
        prev_rgb = dataloader[i - 1]['rgb']
        curr_rgb = dataloader[i]['rgb']
        T_cw, _, _, _, _ = tracker.bootstrap(prev_rgb, curr_rgb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Awesome Visual SLAM")
    parser.add_argument("-c",
                        "--config",
                        default=None,
                        help="Path to the configuration yaml file",
                        type=str)
    parser.add_argument("-f",
                        "--feat",
                        default=None,
                        help="Path to the Feature yaml file",
                        type=str)
    config = ConfigParser.from_args(parser)
    main(config)
