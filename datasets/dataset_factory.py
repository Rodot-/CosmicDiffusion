import typing

from datasets.datasets import RandomData, hesma, hesma_label, tardis_spectra

MODULES = {
    "RandomData": RandomData,
    "hesma": hesma,
    "hesma_label": hesma_label,
    "tardis_spectra": tardis_spectra,
}

# TODO: switch it back when the cluster updated to python 3.8
# NAME_KEY: typing.Final[str] = "name"
# CONFIG_KEY: typing.Final[str] = "config"
NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def get_module(name: str, config: typing.Dict[str, typing.Any]):
    """Recursively deserializes objects registered in MODULES."""
    if name not in MODULES:
        raise KeyError(
            f"{name} not found in registered modules. Available are {MODULES.keys()}."
        )
    cls = MODULES[name]
    for key, value in config.items():
        if isinstance(value, dict) and NAME_KEY in value and CONFIG_KEY in value:
            config[key] = get_module(value[NAME_KEY], value[CONFIG_KEY])
    return cls(**config)
