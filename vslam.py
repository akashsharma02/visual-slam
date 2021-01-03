import numpy as np
import argparse
import tqdm

from vslam.parse_config import ConfigParser
import vslam.dataloaders as loader

def main(config):
    """TODO: Docstring for main.

    :function: TODO
    :returns: TODO

    """
    #TODO: Logger
    dataloader = config.init_obj('dataset', loader)

    for i in tqdm.tqdm(range(len(dataloader)), desc=config['dataset']['type']):
        data = dataloader[i]


if __name__ == "__main__":
    parser      = argparse.ArgumentParser(description="Awesome Visual SLAM")
    parser.add_argument("-c", "--config", default=None, help="Path to the configuration yaml file", type=str)

    config = ConfigParser.from_args(parser)
    main(config)
