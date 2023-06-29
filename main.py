# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""This is the main entry point for the code."""
import os 
os.environ['NUMEXPR_MAX_THREADS'] = '10'

import hydra

from mtrl.app.run import run
from mtrl.utils import config as config_utils
from mtrl.utils.types import ConfigType


@hydra.main(config_path="config", config_name="config")
def launch(config: ConfigType) -> None:
    config = config_utils.process_config(config)
    return run(config) # experiment prepare & run --> set_seed() + experiment.run()


if __name__ == "__main__":
    launch()
