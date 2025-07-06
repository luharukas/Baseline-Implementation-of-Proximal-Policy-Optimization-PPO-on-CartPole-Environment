# Baseline Implementation of Proximal Policy Optimization (PPO) on CartPole-v1

This repository provides a professional-grade implementation of the PPO algorithm applied to the CartPole-v1 environment from the Gymnasium library. The codebase is structured as a Python package with a standalone training script.

## Repository Structure

```
.  
├── README.md
├── requirements.txt
├── train.py
└── ppo_cartpole
    ├── __init__.py
    ├── agent.py
    ├── model.py
    ├── plot.py
    └── utils.py
```

## Installation

```bash
git clone <repo-url>
cd Baseline-Implementation-of-Proximal-Policy-Optimization-PPO-on-CartPole-Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the training script with default hyperparameters:

```bash
python train.py
```

For help on configurable options:

```bash
python train.py --help
```

For example, to train with two hidden layers of sizes 128 and 64:
```bash
python train.py --hidden-dims 128,64
```

## Results

After training, plots for training rewards, testing rewards, and policy/value losses will be displayed.
They are also saved to `train_rewards.png`, `test_rewards.png`, and `losses.png` in the current working directory.