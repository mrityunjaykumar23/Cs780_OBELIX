# OBELIX Reinforcement Learning Project

![OBELIX environment](./OBELIX.png)

This repository contains a reinforcement learning project built around the OBELIX robot simulation. The goal is to train an agent that can navigate the arena, find the box, attach to it, and push it to the boundary while handling progressively harder environment settings.

The project includes:

- the OBELIX simulator
- a rebuilt training pipeline based on discrete Soft Actor-Critic (SAC)
- Random Network Distillation (RND) for exploration
- curriculum learning across multiple difficulty levels
- evaluation scripts for local testing and Codabench-style submission

## Final Project Status

This repository now represents the final consolidated version of the project rather than the original assignment starter state.

The current main implementation is centered around:

- `train_rebuilt.py` as the final training pipeline
- `agent_template.py` as the final inference policy wrapper
- `sac_vec.pth` as the trained actor used for evaluation and submission

Before reaching this version, multiple approaches were tested and stored for reference inside the `submission/` folder. Those archived experiments include earlier submissions based on methods such as Q-learning, NFQ, DQN variants, A2C, PPO, DDQN, and intermediate training scripts from different project phases.

## Project Overview

The environment is inspired by the behaviour-based robot setting described in the paper *Automatic Programming of Behaviour-based Robots using Reinforcement Learning* by Sridhar Mahadevan and Jonathan Connell.

In this implementation:

- the robot receives an 18-dimensional observation vector
- the available actions are `L45`, `L22`, `FW`, `R22`, and `R45`
- an episode is considered successful when the robot attaches to the box and the attached box reaches the arena boundary
- training is performed with parallel environment workers to speed up data collection

## Main Features

- Discrete SAC agent with twin critics
- RND-based intrinsic reward to encourage exploration
- curriculum training over difficulty levels `0`, `2`, and `3`
- multiprocessing-based vectorized environment rollout
- evaluation utilities for both local experiments and benchmark submission
- pretrained actor weights for direct policy inference

## Repository Structure

- `obelix.py`: core environment implementation
- `manual_play.py`: manual control of the robot
- `train_rebuilt.py`: rebuilt training script for the SAC + RND agent
- `evaluate.py`: local evaluation script for a policy file
- `evaluate_on_codabench.py`: evaluation script for benchmark-style submissions
- `agent_template.py`: inference policy that loads trained weights from `sac_vec.pth`
- `compute_observation_states.py`: utility for inspecting observed states
- `sac_vec.pth`: trained actor weights
- `best_sac_vec.pth`: saved best-performing actor snapshot
- `leaderboard.csv`: logged local evaluation results
- `submission/`: archived experimental submissions from earlier project phases
- `result_codabench/`: stored Codabench evaluation outputs from multiple submission rounds
- `Report.tex`: project report source

## Environment and Task

The agent operates in a 2D arena with optional wall obstacles and different box behaviors:

- Difficulty `0`: static box
- Difficulty `2`: blinking / appearing-disappearing box
- Difficulty `3`: moving and blinking box

The objective is not only to reach the box, but to complete the full box-delivery behavior reliably under these settings.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

For training and running the provided neural policy, PyTorch is also required:

```bash
pip install torch
```

## How to Run

### Manual Play

You can control the robot manually with:

```bash
python manual_play.py
```

Controls:

- `w`: move forward
- `a`: turn left by 45 degrees
- `q`: turn left by 22.5 degrees
- `d`: turn right by 45 degrees
- `e`: turn right by 22.5 degrees

### Train the Agent

Run training with:

```bash
python train_rebuilt.py --obelix_py obelix.py
```

Example with explicit settings:

```bash
python train_rebuilt.py --obelix_py obelix.py --episodes 2000 --num_envs 4 --max_steps 1000 --out sac_vec.pth
```

The training script supports:

- curriculum learning
- configurable difficulty and box speed
- parallel rollout workers
- loading pretrained actor weights
- saving the best and final actor checkpoints

### Evaluate a Policy Locally

To evaluate the provided policy template:

```bash
python evaluate.py --agent_file agent_template.py --runs 10 --seed 0 --max_steps 1000 --wall_obstacles
```

This appends a result row to `leaderboard.csv`.

### Benchmark / Submission Evaluation

For benchmark-style evaluation, use:

```bash
python evaluate_on_codabench.py <input_dir> <output_dir>
```

The submission must provide a Python file defining:

```python
def policy(obs, rng) -> str:
    ...
```

## Training Method

The training pipeline in `train_rebuilt.py` combines several ideas:

- a discrete SAC policy for action selection
- twin Q-networks for more stable value estimation
- RND intrinsic reward to improve exploration
- curriculum scheduling so the agent first learns on easier scenarios before harder ones
- vectorized rollout collection using subprocess workers

This setup is designed to improve learning stability and sample efficiency in the OBELIX environment.

## Policy Inference

`agent_template.py` loads the trained actor weights from `sac_vec.pth` and performs greedy action selection on CPU. This makes it suitable for evaluation and submission settings where a lightweight inference-only policy is needed.

## Outputs

During experiments, the repository may produce or update:

- `sac_vec.pth`: final saved actor weights
- `best_sac_vec.pth`: best actor checkpoint during training
- `leaderboard.csv`: local evaluation history
- additional files inside `submission/` or `result_codabench/`

## Experiment History

The project was developed iteratively. Instead of keeping only the final model, this repository also preserves earlier submission attempts and phase-wise experiments.

The `submission/` folder contains archived work from multiple stages, including:

- phase 1 submissions
- phase 2 submissions
- phase 3 submissions
- final submission variants

These folders document the trial-and-error process followed before arriving at the current SAC + RND based final version.

## Codabench Results

The `result_codabench/` folder stores benchmark outputs collected from different evaluation rounds, including:

- `test_phase_result`
- `phase 1 result`
- `phase_2_result`
- `phase_3_result`

These files show the progression of the project across different submissions. In the archived `phase_3_result` runs, the recorded weighted cumulative rewards include:

- `-357558.300`
- `-186346.600`
- `-5837.480`
- `-32129.840`

Among those stored phase 3 runs, the best recorded weighted cumulative reward is `-5837.480`, which indicates a substantial improvement over earlier attempts even though the task remains challenging.

## Notes

- The environment implementation is adapted from an existing OBELIX repository and extended for reinforcement learning experiments.
- The current push behavior is closer to an attach-and-move mechanic than a fully realistic pushing interaction.
- If you publish this repository, avoid committing local-only folders such as `venv/` and `__pycache__/`.

## References

- [Automatic Programming of Behaviour-based Robots using Reinforcement Learning](https://cdn.aaai.org/AAAI/1991/AAAI91-120.pdf)
- [Original OBELIX repository](https://github.com/iabhinavjoshi/OBELIX)
