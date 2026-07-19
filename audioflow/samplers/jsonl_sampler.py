import random

from audioflow.utils.json import read_jsonl


class BatchJsonlSampler:
    r"""Jsonl Sampler.
    """
    def __init__(self, jsonl_paths: list[str], weights: list[float], batch_size: int) -> None:
        self.jsonl_paths = jsonl_paths
        self.weights = weights
        self.batch_size = batch_size

        self.metas = [read_jsonl(path) for path in self.jsonl_paths]  # list[list[str]]
        self.lens = [len(metas) for metas in self.metas]  # list[int]
        self.ptrs = [0 for _ in self.lens]  # list[int]
        self.indices = [self.random_indices(L) for L in self.lens]  # list[list[int]]

    def __iter__(self) -> dict:
        r"""Random sample a jsonl file and sample a line."""

        while True:

            # Randomly sample a jsonl file
            i = random.choices(population=range(len(self.metas)), weights=self.weights, k=1)[0]
            batch_meta = []
            
            for _ in range(self.batch_size):
                # Reset pointer and shuffle indices
                if self.ptrs[i] == len(self.metas[i]):
                    self.indices[i] = self.random_indices(self.lens[i])
                    self.ptrs[i] = 0

                # Randomly sample an item in the jsonl file
                j = self.indices[i][self.ptrs[i]]  # item index
                self.ptrs[i] += 1

                meta = self.metas[i][j]
                batch_meta.append(meta)

            yield batch_meta

    def random_indices(self, N: int) -> list[int]:
        indices = list(range(N))
        random.shuffle(indices)
        return indices