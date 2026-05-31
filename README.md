# TractorBot

**TractorBot** is a Reinforcement Learning agent designed to play the popular Chinese trick-taking card game "Tractor".

## ✨ Features

* **Custom Game Environment**: A fully implemented, lightweight environment for the Tractor card game (`tractorbot.envs`).
* **Distributed RL Architecture**: Features an asynchronous Actor-Learner architecture for efficient training (`tractorbot.rl`).
* **Diverse Baselines**: Includes heuristic-based bots (`heu_bot`) and rule-based bots (`better_bot`) for evaluation and baseline comparisons.

## 📂 Project Structure

```text
TractorBot/
├── pretrained_models/          # Pretrained model checkpoints (e.g., model_phase1.pt)
├── scripts/                    # Executable scripts for training and evaluation
│   ├── train.py                # Main RL training script
│   ├── heu_bot_train.py        # Heuristic bot training/evaluation
│   └── play.py                 # Interactive testing / evaluation script
├── src/
│   └── tractorbot/             # Core Python package
│       ├── envs/               # Game environment and state representations
│       ├── agents/             # Bot implementations (heuristic, rule-based)
│       ├── rl/                 # Reinforcement learning core (actor, learner, replay buffer)
│       └── models/             # Neural network architectures
├── pyproject.toml              # Project metadata and build configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file

```

## Installation

**1. Clone the repository:**

```bash
git clone https://github.com/yourusername/TractorBot.git
cd TractorBot
pip install -r requirements.txt
```

**2. Install the project in editable mode:**
This allows you to import `tractorbot` from anywhere in the project.

```bash
pip install -e .
```

## Training

To start a new Reinforcement Learning training loop using the Actor-Learner framework:

```bash
python scripts/train.py
```

## Pretrained Models

Pretrained weights are located in the `pretrained_models/` directory.

## Contributing

Contributions are welcome! If you have ideas for improving the environment, adding new RL algorithms, or optimizing the neural network architecture, please feel free to:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

Developed as part of the PKU Reinforcement Learning Final Project.