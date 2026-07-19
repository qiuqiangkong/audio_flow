import yaml


def read_yaml(path) -> dict:
    r"""Parse yaml file."""
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)