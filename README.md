# Normalizing Flow for Physics and Mathematical Targets

This repository contains a modular PyTorch implementation of Normalizing Flows using Affine Coupling Layers. These models are designed to learn complex analytical target PDFs such as the Breit-Wigner resonance (Z-boson physics) and the 8-dimensional Rosenbrock function.

## Overview
This project refactors the original Jupyter notebook models into a structured, modular Python package that can be easily extended and version-controlled.

## Installation

You can install the dependencies via pip:

```bash
pip install -r requirements.txt
```

Ensure your Python environment supports PyTorch and matplotlib.

## Usage

The main logic resides in `src/normalizing_flow/`. The Jupyter notebooks that demonstrate step-by-step developments and math fundamentals have been preserved in the `notebooks/` directory.

To test the package, you can run the test script:

```bash
python scripts/test_run.py
```

## Structure

- `src/normalizing_flow/models.py`: AffineCouplingLayer and NormalizingFlow.
- `src/normalizing_flow/targets.py`: Physics target functions.
- `src/normalizing_flow/train.py`: The adaptive training loop with Early Stopping and Cosine Annealing.
- `src/normalizing_flow/utils.py`: Integration and rendering plots for the learned distributions.
- `notebooks/`: Collection of experimental standard notebooks.
