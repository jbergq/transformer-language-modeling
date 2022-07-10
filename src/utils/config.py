import yaml

from .easy_dict import EasyDict


def load_config(path):
    with open(path, "r") as stream:
        try:
            return EasyDict(yaml.safe_load(stream))
        except yaml.YAMLError as e:
            print(e)
