from tractorbot.envs.env import TractorEnv
from tractorbot.models.model import CNNModel
from tractorbot.config import build_training_config, load_config


def test_environment_reset_returns_observation_and_actions():
    env = TractorEnv({"seed": 0})

    observation, actions = env.reset(level="2", banker_pos=0, major="s")

    assert observation["id"] == 0
    assert len(observation["deck"]) == 25
    assert isinstance(actions, list)
    assert actions


def test_model_can_be_constructed():
    model = CNNModel()

    assert model is not None


def test_default_config_builds_training_config():
    config = build_training_config(load_config())

    assert config["model_pool_name"] == "tractor_pool"
    assert config["batch_size"] > 0
    assert "env" in config
    assert "opponents" in config
