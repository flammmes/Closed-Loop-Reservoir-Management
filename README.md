# Closed-Loop Reservoir Management

Python code for closed-loop CO2 storage control using deep reinforcement learning and latent model-based adaptation.

## Overview

This repository contains code for closed-loop reservoir management experiments for geological CO2 storage. The workflow formulates CO2 injection and brine-production control as a partially observable sequential decision problem, where controllers act from realistically available well-level observations rather than full reservoir states.

The main research direction is to train deployable reinforcement-learning controllers with high-fidelity reservoir simulation and then adapt them under abnormal operating scenarios using latent model-based retuning.

## Paper context

The associated work studies closed-loop CO2 storage control with history-based reinforcement learning and latent model-based adaptation. The control problem is modeled as a partially observable Markov decision process, where the full reservoir state is latent and the deployed controller relies on well-level observations and reward signals.

The experiments compare several model-free information regimes:

- privileged-state SAC using dense simulator fields and well observations;
- well-only SAC using only current well responses;
- history-conditioned SAC using a rolling sequence of well observations;
- masked-critic curriculum learning;
- asymmetric teacher-student learning with privileged critic supervision.

The model-based part maps public well-history observations to a latent control state, trains latent dynamics models, and performs Dreamer-style imagined rollouts for controller adaptation.

## Main ideas

- Closed-loop control for geological CO2 storage.
- Partial observability with deployable well-level observations.
- History-conditioned policies using finite well-response histories.
- Soft Actor-Critic controllers for continuous well controls.
- Teacher-student training with privileged simulator-state critics.
- Latent model-based adaptation under abnormal operating conditions.
- Scenario-specific retuning with limited high-fidelity simulator interaction.

## Abnormal scenarios

The code supports experiments motivated by:

1. known injector-control loss;
2. leakage-induced dynamics and reward shift;
3. compartmentalized reservoir connectivity shift.

## Repository contents

This repository intentionally contains only Python source files and lightweight metadata.

Large simulator outputs, reservoir decks, trained checkpoints, environment files, private keys, and machine-specific files are excluded.

## Requirements

The main dependencies are expected to include:

- Python 3.11
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Gymnasium
- Tianshou
- OPM Flow / reservoir-simulation tooling, where applicable

Install the exact environment according to your local simulator setup.

## Notes

This is a research-code release. Paths, simulator decks, and machine-specific configuration may need to be adapted before running on a new system.
