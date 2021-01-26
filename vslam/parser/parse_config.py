from __future__ import annotations
import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
import argparse
import yaml

from .cfgnode import CfgNode


class ConfigParser:
    def __init__(self, config) -> None:
        self.config = CfgNode(config)

        self.save_dir = Path(self.config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Update configuration with new attributes if added
        with open(self.save_dir / 'config.yaml', 'w') as updated_cfg_file:
            yaml.dump(self.config, updated_cfg_file)


    @classmethod
    def from_args(cls: ConfigParser, args: argparse.ArgumentParser) -> ConfigParser:
        """
        Initialize ConfigParser from command line arguments and config
        """
        if not isinstance(args, tuple):
            args = args.parse_args()

        msg_no_cfg = "Configuration file needs to be specified. Add '-c config.yaml' argument when running"
        assert args.config is not None, msg_no_cfg
        cfg_filename = Path(args.config)

        filestream = open(cfg_filename, 'r')
        config = yaml.load(filestream, Loader=yaml.FullLoader)
        return cls(config)


    def init_obj(self, name, module, *args, **kwargs):
        """

        Finds a function handle with the name given as 'type' in the config, and returns an instance
        initialized with corresponding arguments given
        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`

        """
        # module_name = self[name]['type']
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)


    def __getattr__(self, name: str):
        """Access items with keys"""
        if name in self.config:
            return self.config[name]
        else:
            raise AttributeError(name)


    def __getitem__(self, name):
        """Access items like ordinary dict."""
        if name in self.config:
            return self.config[name]
        else:
            raise AttributeError(name)
