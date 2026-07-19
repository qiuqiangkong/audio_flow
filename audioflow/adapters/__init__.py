import torch.nn as nn


def get_adapter(configs: dict) -> nn.Module:
    r"""Initialize adapter."""
    name = configs["name"]

    if name in ["TTAAdapter"]:
        from .tta import TTAAdapter
        return TTAAdapter(**configs)

    elif name in ["TTSAdapter"]:
        from .tts import TTSAdapter
        return TTSAdapter(**configs)

    elif name in ["Any2AudioAdapter"]:
        from .any2audio import Any2AudioAdapter
        return Any2AudioAdapter(**configs)

    elif name in ["Token2AudioAdapter"]:
        from .token2audio import Token2AudioAdapter
        return Token2AudioAdapter(**configs)

    else:
        raise ValueError(name)