import argparse
from tqdm import tqdm
import cv2

from vslam.parser import ConfigParser, CfgNode
from vslam import visualizer as viz
from vslam import dataloaders
from vslam import feature
from vslam.types import PinholeCamera, Frame, Map
from vslam.tracker import Tracker

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

    prev_data = dataloader[0]
    curr_data = dataloader[1]

    feature_extractor = config.init_obj('feature', feature)
    tracker = Tracker(config.tracker.args, config.map.args, camera)
    slam_map = None

    print(tracker.state)
    for i in tqdm(range(0, 10), desc=config.dataset.type):
        data = dataloader[i]
        curr_frame = Frame(data['rgb'], data['rgb_timestamp'], config.frame.args, camera, feature_extractor)
        tracker.track(curr_frame, slam_map)
        print(tracker.state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Awesome Visual SLAM")
    parser.add_argument("-c",
                        "--config",
                        default=None,
                        help="Path to the configuration yaml file",
                        type=str)
    config = ConfigParser.from_args(parser)
    main(config)
