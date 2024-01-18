from omegaconf import DictConfig
from hydra import compose, initialize


def compose_config(
    config_name: str, base_config_path: str = "../configs"
) -> DictConfig:
    with initialize(config_path=base_config_path, version_base="1.1"):
        return compose(config_name=config_name)
