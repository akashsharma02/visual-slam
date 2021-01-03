import numpy as np
import argparse
import tqdm
import cv2

from vslam.parse_config import ConfigParser
from vslam import visualizer as viz
from vslam import dataloaders

def main(config):
    """TODO: Docstring for main.

    :function: TODO
    :returns: TODO

    """
    #TODO: Logger
    dataloader = config.init_obj('dataset', dataloaders)

    prev_data = dataloader[0]
    print(len(dataloader))
    for i in tqdm.tqdm(range(1, len(dataloader)), desc=config['dataset']['type']):
        data = dataloader[i]

        cv2.imshow("rgb image", data["rgb"])
        cv2.waitKey(1)


if __name__ == "__main__":
    parser      = argparse.ArgumentParser(description="Awesome Visual SLAM")
    parser.add_argument("-c", "--config", default=None, help="Path to the configuration yaml file", type=str)

    config = ConfigParser.from_args(parser)
    main(config)
