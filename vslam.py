import argparse
from tqdm import tqdm
import cv2

from vslam.parser import ConfigParser, CfgNode
from vslam import visualizer as viz
from vslam import dataloaders
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

    # prev_frame = Frame(prev_data['rgb'], prev_data['timestamp'], config.frame.args, camera)

    #curr_frame = Frame(curr_data['rgb'], curr_data['timestamp'], camera)
    #prev_rgb, curr_rgb = prev_data["rgb"], curr_data["rgb"]


    tracker = Tracker(config.tracker.args, config.map.args, camera)
    slam_map = None
    ##TODO: Change interface to accept camera
    ## T_cw, P = tracker.bootstrap(prev_frame, curr_frame)
    ## print(P)

    print(tracker.state)
    for i in tqdm(range(0, 5), desc=config.dataset.type):
        data = dataloader[i]
        curr_frame = Frame(data['rgb'], data['timestamp'], config.frame.args, camera)
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
