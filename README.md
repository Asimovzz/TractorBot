# TractorBot

TractorBot is a PyTorch reinforcement-learning project for the Chinese
trick-taking card game Tractor. It includes a playable game environment,
heuristic baseline bots, neural-network models, distributed self-play training
components, pretrained checkpoints, and an online-evaluation style play script.

## Current Status

This repository is research-oriented and currently marked as alpha. The core
environment and training pipeline are available, but public APIs may still
change as the project is cleaned up and tested more broadly.

## Features

- Tractor game environment with legal move generation, scoring, reward shaping,
  and action validation.
- Baseline agents, including rule-based and heuristic bots.
- PyTorch CNN actor-critic model for action scoring and value estimation.
- Actor-learner reinforcement-learning components with replay buffer and model
  pool support.
- Pretrained checkpoints in `pretrained_models/`.
- Botzone-style `scripts/play.py` entry point that can fall back to heuristic
  play when a neural checkpoint is unavailable.

## Repository Layout

```text
.
|-- configs/                 # Experiment and environment configuration
|   `-- default.yaml
|-- pretrained_models/       # Published pretrained checkpoints
|-- scripts/                 # Runnable training and play entry points
|   |-- train.py
|   `-- play.py
|-- src/tractorbot/          # Python package
|   |-- agents/              # Rule and heuristic bots
|   |-- envs/                # Game environment, wrappers, move generation
|   |-- models/              # Neural network architectures
|   `-- rl/                  # Actor, learner, replay buffer, model pool
|-- tests/                   # Minimal smoke tests
|-- CHANGELOG.md
|-- CONTRIBUTING.md
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
`-- SECURITY.md
```

## Installation

Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick Start

Create an environment and inspect the first observation:

```python
from tractorbot.envs.env import TractorEnv

env = TractorEnv({"seed": 0})
obs, actions = env.reset(level="2", banker_pos=0, major="s")

print(obs["id"])
print(obs["deck"][:5])
print(actions[:3])
```

Run the training entry point:

```bash
python scripts/train.py
```

Training defaults are loaded from `configs/default.yaml`. To run with another
configuration file:

```bash
python scripts/train.py --config path/to/config.yaml
```

Run the play script:

```bash
python scripts/play.py
```

When run locally, `scripts/play.py` reads `log_forAI.json` if present and prints
a JSON response. In an online judge environment, it reads a single JSON request
from standard input.

## Pretrained Models

The repository includes:

- `pretrained_models/model_phase1.pt`
- `pretrained_models/model_phase2.pt`

These checkpoints are referenced from `configs/default.yaml`. The Botzone-style
play script currently looks for `/data/model_plus_global.pt` in online mode and
falls back to heuristic decisions if that file is unavailable.

## Configuration

`configs/default.yaml` documents environment rules, reward constants,
reinforcement-learning settings, opponent sampling weights, and model settings.
Both `scripts/train.py` and `scripts/play.py` accept `--config`. You can also set
`TRACTORBOT_CONFIG` to point at a YAML file.

## Contributing

See `CONTRIBUTING.md` for setup and pull-request guidance. Bug reports that
include a minimal game state or reproduction script are especially useful.

## License

TractorBot is distributed under the MIT License. See `LICENSE` for details.
