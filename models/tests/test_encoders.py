import typing

import pytest

from models.encoders import BaseEncoder, MLPEncoder

OBSERVATION_DIM = 10


def test_base_encoder() -> None:
    enc = BaseEncoder(OBSERVATION_DIM)
    assert enc.observation_dim == OBSERVATION_DIM


@pytest.mark.parametrize(
    "config",
    [
        {"observation_dim": OBSERVATION_DIM},
        {"observation_dim": OBSERVATION_DIM, "hidden_dims": (20,)},
    ],
)
def test_mlp_encoder(config: typing.Dict[str, typing.Any]) -> None:
    enc = MLPEncoder(**config)
    for k, v in config.items():
        assert getattr(enc, k) == v
