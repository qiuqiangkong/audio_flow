from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR

from .schedulers import LinearWarmUp


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[Tensor]
) -> tuple[Optimizer, LambdaLR | None]:
    r"""Get optimizer and scheduler."""

    optimizer_name = configs["name"]
    lr = float(configs["lr"])
    warm_up_steps = configs["warmup_steps"]

    if optimizer_name == "AdamW":
        optimizer = AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler