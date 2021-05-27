from argparse import Namespace
import yaml
import os
from typing import Dict


def write_config(config_path: str, config: Dict):
    dirname = os.path.dirname(config_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(config_path, "w") as writer:
        yaml.dump(config, writer)