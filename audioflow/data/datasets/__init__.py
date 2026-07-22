from torch.utils.data import Dataset


def get_dataset(configs: dict) -> Dataset:
    r"""Get dataset.
    """
    name = configs["name"]

    if name in ["TTADataset"]:
        from .tta import TTADataset
        return TTADataset(configs["crop_duration"])

    elif name in ["Any2AudioDataset"]:
        from .any2audio import Any2AudioDataset
        return Any2AudioDataset(configs["crop_duration"])

    else:
        raise ValueError(name)
