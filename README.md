# Self-Growing Machine (Programmable NCA)

This project implements a **Growing Neural Cellular Automaton (NCA)** that achieves high levels of open-ended self-organization and robust adaptation.

## Achievements

1.  **Pattern Growth**: Starting from a tiny seed, the system grows into a complex target structure.
2.  **Self-Maintenance**: The structure remains stable over time after growth.
3.  **Self-Healing**: The organism can recover from significant damage (e.g., punched holes) automatically.
4.  **Robust Adaptation**: The system continues to function and repair itself even under stochastic state noise.
5.  **Open-ended Morphogenesis**: The **same local rule** can grow multiple different structures (critter, square, circle, triangle) depending on a "DNA" signal injected into the seed cell's hidden channels.

## Core Components

- `nca_core_local_engine.py`: The core engine containing the `UpdateRule` (the neural net), `Perception` (local sensing), and the training loop.
- `train_programmable_nca.py`: Script to train the model on multiple targets with robustness noise.
- `verify_achievements.py`: Demonstrates the model's capabilities in growth, multi-shape morphogenesis, and healing.
- `nca_kaggle.ipynb`: Original notebook for high-scale GPU training.

## How it works

The "Update Rule" is a tiny MLP (1x1 convolutions) replicated at every pixel. Each cell senses its 3x3 neighborhood through a fixed perception layer (identity + Sobel filters). Information propagates only through local interactions.

Programmability is achieved by reserving hidden channels in the seed for a "DNA" vector. The cells learn to interpret this local signal to determine which global morphology to build.

## Results

- `achievement_morphogenesis.png`: Shows different shapes grown from the same rule.
- `achievement_robustness.png`: Shows damage recovery.
