import hydra

from mtrl.utils import config as config_utils
from mtrl.utils.types import ConfigType


# TODO: load best model and record the video


@hydra.main(config_path="config", config_name="config")
def launch(config: ConfigType) -> None:
    config = config_utils.process_config(config)
    config_utils.pretty_print(config, resolve=False)
    # return run(config) 
    # TODO prepare a class like Experiment that simulate & record video


if __name__ == "__main__":
    launch()