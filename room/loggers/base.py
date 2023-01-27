from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Dict, Generator, List, Mapping, MutableMapping, Optional, Tuple, Union


def flatten_dict(params: Dict[Any, Any], delimiter: str = "/") -> Dict[str, Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.
    Returns:
        Flattened dict.
    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(
        input_dict: Any, prefixes: List[Optional[str]] = None
    ) -> Generator[Any, Optional[List[str]], List[Any]]:
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (MutableMapping, Namespace)):
                    value = vars(value) if isinstance(value, Namespace) else value
                    yield from _dict_generator(value, prefixes + [key])
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


class Logger(ABC):
    @abstractmethod
    def log_hparams(self, **kwargs):
        pass

    @abstractmethod
    def log_metrics(self, **kwargs):
        pass
