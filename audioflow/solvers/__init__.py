def get_solver(configs: dict) -> callable:
    
    name = configs["name"]

    if name == "euler":
        from .euler import euler_solver
        return euler_solver

    else:
        raise ValueError(name)
