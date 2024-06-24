from models.decoders import Label_MLPDecoder, MLPDecoder
from models.encoders import Label_MLPEncoder, MLPEncoder
from models.optimizer import Adam, RAdam, Ranger
from models.scheduler import (
    AnnealingLinearScheduler,
    ConstantScheduler,
    ConstrainedExponentialSchedulerMaLagrange,
)
from models.spectraNN import spectraNN
from models.spectraNN_label import spectraNNLabel
from models.vae_adversarial import VAEAdversarial
from models.vae_fm import VAEFM
from models.vae_label import VAELabel
from models.vae_mixup import VAEMixup

MODULES = {
    "MLPEncoder": MLPEncoder,
    "MLPDecoder": MLPDecoder,
    "AnnealingLinearScheduler": AnnealingLinearScheduler,
    "ConstrainedExponentialSchedulerMaLagrange": ConstrainedExponentialSchedulerMaLagrange,
    "RAdam": RAdam,
    "Ranger": Ranger,
    "Adam": Adam,
    "Label_MLPEncoder": Label_MLPEncoder,
    "Label_MLPDecoder": Label_MLPDecoder,
    "VAELabel": VAELabel,
    "ConstantScheduler": ConstantScheduler,
    "SpectraNN": spectraNN,
    "SpectraNNLabel": spectraNNLabel,
    "VAEAdversarial": VAEAdversarial,
    "VAEFM": VAEFM,
    "VAEMixup": VAEMixup,
}

# TODO: switch it back when the cluster updated to python 3.8
# NAME_KEY: typing.Final[str] = "name"
# CONFIG_KEY: typing.Final[str] = "config"
NAME_KEY: str = "name"
CONFIG_KEY: str = "config"


def get_module(name: str, config) -> VAELabel:
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
