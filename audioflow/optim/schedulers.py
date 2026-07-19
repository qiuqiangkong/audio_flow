class LinearWarmUp:
    r"""Linear learning rate warm up scheduler."""

    def __init__(self, warmup_steps: int) -> None:
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        if step <= self.warmup_steps:
            return step / self.warmup_steps
        else:
            return 1.