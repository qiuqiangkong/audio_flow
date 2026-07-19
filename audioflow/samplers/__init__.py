from torch.utils.data import Sampler


def get_batch_sampler(configs: dict) -> Sampler:
    r"""Get sampler.
    """
    name = configs["sampler"]["name"]
    batch_size = configs["train"]["batch_size_per_device"]

    if name == "BatchJsonlSampler":
        from audioflow.samplers.jsonl_sampler import BatchJsonlSampler
        paths = [meta["path"] for meta in configs["train_jsonls"]]
        weights = [meta["weight"] for meta in configs["train_jsonls"]]
        return BatchJsonlSampler(paths, weights, batch_size)

    else:
        raise ValueError(name)