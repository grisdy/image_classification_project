import yaml
from src.constant import Path


def get_config():
    with open(Path.CONFIG, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf
    