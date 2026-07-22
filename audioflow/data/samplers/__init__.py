from torch.utils.data import Sampler


def get_batch_sampler(configs: dict) -> Sampler:
    r"""Get sampler.
    """
    
    name = configs["data"]["sampler"]["name"]

    if name == "BatchJsonlSampler":
        from .jsonl_sampler import BatchJsonlSampler
        paths = [meta["path"] for meta in configs["data"]["train"]]
        weights = [meta["weight"] for meta in configs["data"]["train"]]
        batch_size = configs["train"]["batch_size_per_device"]
        return BatchJsonlSampler(paths, weights, batch_size)

    else:
        raise ValueError(name)