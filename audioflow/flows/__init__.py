def get_flow(configs):
    t_schedule = get_t_schedule(configs["t_schedule"])
    fm = get_matcher(configs["matcher"], t_schedule)
    return fm


def get_t_schedule(configs):
    name = configs["name"]
    
    if name == "uniform":
        from .t_schedules import Uniform
        return Uniform()

    elif name == "logitnormal":
        from .t_schedules import LogitNormal
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