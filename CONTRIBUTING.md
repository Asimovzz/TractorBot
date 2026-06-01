# Contributing

Thanks for taking the time to improve TractorBot.

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Before Opening a Pull Request

Run the smoke tests:

```bash
python -m pytest
```

For changes to game rules, action generation, reward shaping, or model inputs,
include a short explanation of the gameplay scenario being changed and add a
focused test when practical.

## Project Conventions

- Keep public examples runnable from a fresh editable install.
- Prefer small, reviewable changes over broad refactors.
- Do not commit local training outputs, logs, or experimental checkpoints unless
  they are intentionally published as pretrained artifacts.
- Document any compatibility break in `CHANGELOG.md`.
