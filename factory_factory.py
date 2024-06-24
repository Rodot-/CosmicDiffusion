import typing
from inspect import signature

from datasets.dataset_factory import MODULES as DATASET_MODULES
from experiments.factory import MODULES as EXPERIMENT_MODULES
from models.factory import MODULES as MODEL_MODULES

MODULE_DIR = {
    "dataset": DATASET_MODULES,
    "model_params": MODEL_MODULES,
    "exp_params": EXPERIMENT_MODULES,
}


NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def parse_config(config: typing.Dict):
    outputs = {}
    for section, sub_config in config.items():
        if section not in MODULE_DIR:
            continue
        MODULES = MODULE_DIR[section]
        if (
            NAME_KEY in sub_config and CONFIG_KEY in sub_config
        ):  # Then it is a module we can parse
            name = sub_config[NAME_KEY]
            outputs[section] = get_module(name, sub_config[CONFIG_KEY], MODULES)
    return outputs


def get_module(name: str, config, MODULES) -> VAELabel:
    """Recursively deserializes objects registered in MODULES."""
    if name not in MODULES:
        raise KeyError(
            f"{name} not found in registered modules. Available are {MODULES.keys()}."
        )
    cls = MODULES[name]
    sig = signature(cls.__init__)
    for key, value in config.items():

        key_sig = sig.parameters[
            key
        ].annotation  # Check that we're not supposed to pass a dict!
        if hasattr(key_sig, "__origin__"):
            key_sig = key_sig.__origin__
        if isinstance(value, key_sig):
            pass

        elif isinstance(value, dict) and NAME_KEY in value and CONFIG_KEY in value:
            config[key] = get_module(value[NAME_KEY], value[CONFIG_KEY])
    return cls(**config)
