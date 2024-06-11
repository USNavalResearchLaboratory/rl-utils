import importlib
import os
import sys
from datetime import datetime
from typing import Any

from gymnasium.envs.registration import EnvSpec, WrapperSpec

PathOrStr = str | os.PathLike


def get_now():
    """Get current date/time in ISO format."""
    str_ = datetime.now().isoformat(timespec="seconds")
    if sys.platform.startswith("win32"):
        str_ = str_.replace(":", "_")
    return str_


def _load_attr(entry_point: str) -> Any:
    """Load module attribute from string specification.

    Assumes `entry_point` is of the form "<module>:<attr>", where <module> is an
    importable module or subpackage (e.g. "foo.bar") and <attr> is the attribute to
    load from the module.

    Note:
        Inspired by `gymnasium.envs.registration.load_env_creator`.

    Args:
        entry_point: The string specification "<module>:<attr>"

    Returns:
        The desired attribute, as if `from <module> import <attr>` had been executed.

    """
    # _prefix = "import::"
    # if entry_point.startswith(_prefix):
    #     entry_point = entry_point.removeprefix(_prefix)
    # else:
    #     raise ValueError
    if ":" not in entry_point:
        raise ValueError
    mod_name, attr_name = entry_point.split(":")
    if mod_name.endswith(".py"):
        name = mod_name.removesuffix(".py").replace("/", ".")
        spec = importlib.util.spec_from_file_location(name, mod_name)
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        sys.modules["module.name"] = mod
        spec.loader.exec_module(mod)  # type: ignore
    else:
        mod = importlib.import_module(mod_name)
    return getattr(mod, attr_name)


def _load_and_construct(obj):
    if isinstance(obj, dict):
        obj = {k: _load_and_construct(v) for k, v in obj.items()}

        if "entry_point" in obj.keys():  # construct object
            entry_point = obj["entry_point"]
            args, kwargs = obj.get("args", ()), obj.get("kwargs", {})
            obj = entry_point(*args, **kwargs)

        return obj
    elif isinstance(obj, list):
        return list(map(_load_and_construct, obj))
    elif isinstance(obj, str):
        try:
            return _load_attr(obj)
        except ValueError:
            return obj
    else:
        return obj


def _make_env_spec(cfg: dict):
    def _fn(w):
        w["kwargs"] = _load_and_construct(w.pop("kwargs"))
        return WrapperSpec(**w)

    cfg["additional_wrappers"] = tuple(map(_fn, cfg["additional_wrappers"]))
    return EnvSpec(**cfg)


# TODO: better separate model cfg, algo vs learning
# TODO: rework CLI parsing w/ defaults and `Namespace`?
