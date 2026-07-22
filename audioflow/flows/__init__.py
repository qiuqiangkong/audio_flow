def get_flow(configs: dict):
    time_sampler = get_time_sampler(configs["time_sampler"])
    fm = get_matcher(configs["matcher"], time_sampler)
    return fm


def get_time_sampler(configs: dict):
    name = configs["name"]
    
    if name == "uniform":
        from .time_samplers import Uniform
        return Uniform()

    elif name == "logitnormal":
        from .time_samplers import LogitNormal
        return LogitNormal(configs["mu"], configs["sigma"])

    else:
        raise ValueError(name)


def get_matcher(configs, t_schedule):

    name = configs["name"]

    if name == "linear":
        from .matchers import FlowMatcher
        return FlowMatcher(t_schedule)

    else:
        raise ValueError(name)