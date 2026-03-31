# Normalizing Flows for Variance Reduction in Monte Carlo Integration

This repository contains a modular PyTorch implementation of **Normalizing Flows** based on **Affine Coupling Layers**. The main goal is not only density modelling, but also **variance reduction for Monte Carlo integration** by learning an efficient **importance-sampling distribution** for difficult target functions.

In physics applications, Monte Carlo integrals often become inefficient when the integrand has sharp peaks, narrow resonances, non-trivial boundaries, or strong interference structures. A normalizing flow can be trained to approximate the shape of such an integrand, so that sampling is concentrated in the regions that matter most. This can significantly reduce the variance of the Monte Carlo weights and therefore improve the statistical precision of the integral estimate.

## Motivation

Many target functions relevant in particle physics are highly structured and are badly matched by simple proposal distributions such as uniform or Gaussian sampling. This is especially true for:

- Breit-Wigner resonances,
- interference patterns between several amplitudes,
- curved or constrained phase-space regions,
- higher-dimensional benchmark functions such as Rosenbrock-type targets.

The idea in this project is to use a normalizing flow as a **learned proposal distribution** $q(x)$ for importance sampling. Instead of drawing samples from a naive distribution, the flow learns to generate samples in the important regions of the target $f(x)$.

The integral is then estimated through importance weights $w(x) = \frac{f(x)}{q(x)}$.


The exact value of the integral does not change, but a better proposal $q(x)$ leads to a smaller variance of the weights $\mathrm{Var}(w)$ and therefore to a smaller Monte Carlo error.

---

## What this repository does

This project refactors earlier notebook-based flow models into a more structured and reusable Python package. It provides:

- a modular implementation of **Affine Coupling normalizing flows** in PyTorch,
- training utilities for learning difficult target densities,
- example target functions from physics and mathematics,
- tools for evaluating the learned proposal via Monte Carlo integration,
- notebooks that preserve the original exploratory development and derivations.

The repository is therefore useful both as:
1. a research prototype for **importance sampling in physics**, and  
2. a clean starting point for extending normalizing flows to other integrands and phase-space problems.

---

## Example physics test case

One of the main toy examples in this project is a **5-dimensional resonance model** inspired by particle-physics phase-space variables.

In this example, the flow learns a target distribution built from:

- **Dalitz-like kinematics** with constrained invariant masses,
- several bounded physical variables obtained from unconstrained neural-network coordinates,
- the associated **Jacobian factors** from the coordinate transformation,
- a matrix-element-like structure consisting of **five Breit-Wigner resonances**,
- non-trivial **interference effects** through complex amplitudes and angular dependence.

Concretely, the model starts from latent Gaussian variables $z \sim \mathcal{N}(0, I)$, transforms them through the normalizing flow, and interprets the output as unconstrained variables $y$. These are then mapped into physical variables such as $s_{12}$, $s_{23}$, angular variables, and phases through sigmoid-based bounded transformations. The full target density includes both:

1. the transformed “physics” part (for example the squared matrix element), and  
2. the corresponding **log-Jacobian** from the bounded variable transformation.

This makes the example more realistic than a simple synthetic benchmark, because it combines:
- sharp resonant structures,
- kinematic constraints,
- angular dependence,
- and multimodal/interfering behaviour.

---

## Method in words

The workflow is:

1. Sample latent variables $z$ from a simple base Gaussian.
2. Push them through the normalizing flow to obtain samples $x$ from the learned proposal.
3. Evaluate the learned proposal density $q(x)$ using the change-of-variables formula.
4. Evaluate the target log-density $\log f(x)$.
5. Train the flow so that $q(x)$ increasingly resembles the target shape.
6. Use the trained flow as an importance sampler for Monte Carlo integration.

The key benefit is that the flow learns where the important regions of the integrand are, so fewer samples are wasted in irrelevant parts of the domain.

---

## Training strategy

The training loop is designed for stability on difficult targets. In the current setup, it includes:

- **Affine Coupling Layers** with alternating binary masks,
- **identity-like initialization** of the final coupling outputs to avoid unstable early transformations,
- **LeakyReLU** hidden networks,
- **Adam** optimization,
- **beta annealing** so that the target is introduced gradually,
- **cosine annealing learning-rate scheduling**,
- **gradient clipping** to control exploding gradients,
- **early stopping** once the loss has plateaued.

This is particularly useful for sharply peaked targets, where fully aggressive training from the beginning can easily become unstable.

The training objective effectively encourages the proposal distribution $q(x)$ to move toward the target shape. In practice, this means the flow becomes better at placing probability mass near resonances and other dominant regions of the integrand.

---

## Monte Carlo integration

After training, the flow is used for importance sampling. Given samples from the learned proposal, the integral is estimated as $I = \int f(x) dx \approx \frac{1}{N}\sum_{i=1}^N \frac{f(x_i)}{q(x_i)}$.


The repository also computes:

- the estimated integral,
- the variance of the importance weights,
- the corresponding Monte Carlo error,
- and visualizations of the sampled distribution.

For a good learned proposal, the weights fluctuate less strongly than with naive sampling, which is exactly what reduces the final integration uncertainty.

---

## Overview

This project refactors the original Jupyter notebook models into a structured, modular Python package that can be easily extended and version-controlled.

## Installation

You can install the dependencies via pip:

```bash
pip install -r requirements.txt
```

## Structure 
```
- src/normalizing_flow/models.py: AffineCouplingLayer and NormalizingFlow.
- src/normalizing_flow/targets.py: Physics target functions.
- src/normalizing_flow/train.py: The adaptive training loop with Early Stopping and Cosine Annealing.
- src/normalizing_flow/utils.py: Integration and rendering plots for the learned distributions.
- notebooks/: Collection of experimental standard notebooks.
```
