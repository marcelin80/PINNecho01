"""Config load/save round-trips and default integrity."""

from pinnecho.config import Config, load_config, config_from_dict


def test_defaults_construct():
    c = Config()
    assert c.physics.backbone == "baseline"
    assert c.train.weights.wall > 0


def test_yaml_roundtrip(tmp_path):
    c = Config()
    c.physics.backbone = "fsi_informed"
    c.train.iterations = 123
    path = tmp_path / "c.yaml"
    c.save(path)
    c2 = load_config(path)
    assert c2.physics.backbone == "fsi_informed"
    assert c2.train.iterations == 123
    assert c2.geometry.r0_y == c.geometry.r0_y


def test_unknown_key_raises():
    import pytest

    with pytest.raises(KeyError):
        config_from_dict({"not_a_key": 1})


def test_tuple_fields_parsed():
    c = config_from_dict({"doppler": {"transducer": [0.1, -0.2]}})
    assert c.doppler.transducer == (0.1, -0.2)
