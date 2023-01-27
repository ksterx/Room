from room import log


def set_param(param: str, value, cfg: dict):
    try:
        if cfg[param] is None:
            log.info(f"Parameter {param} set to {value}")
            cfg[param] = value
        elif cfg[param] is not None:
            log.warning(f"Parameter {param} overriden from {cfg[param]} to {value}")
            cfg[param] = value
            # TODO: log new value with logger
        elif value is None:
            raise ValueError(f"Parameter {param} is not set")
    except KeyError:
        log.critical(f"Parameter {param} not found in config")
