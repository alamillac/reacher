[//]: # (Image References)

[image1]: https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif "Trained Agent"

# Project 2: Reacher

## Introduction

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Getting Started

### Requirements
- Python 3.6 or later
- Conda

### Installation
1. Clone the repository
2. Create a conda environment
```bash
conda create --name drlnd python=3.11
```
3. Activate the environment
```bash
conda activate drlnd
```
4. Install the required packages
```bash
pip install -r requirements.txt
```
5. Install the requirements from the `python` folder
```bash
cd python
pip install .
```
6. Download the Unity environments from one of the links below. You need only select the environment that matches your operating system:

#### Version 1: One (1) Agent

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

#### Version 2: Twenty (20) Agents

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

7. Unzip the file and place it in the root directory of the repository

### (Optional) Crawl

The goal is to teach a creature with four legs to walk forward without falling.

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).

You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Unzip the file and place it in the root directory of the repository

### Testing the Agent

There are already pre-trained weights in the `checkpoint.pth` and `checkpoint_visual.pth` file for 2000 episodes.

To test the agent, run the `test.py` script.

```bash
python src/test.py
```

### Training the Agent

To train the agent, run the `train.py` script.

```bash
python src/train.py
```

## Implementation

In this project, I used DDPG and TD3 algorithms with Priority Experience Replay (PER) to solve the environment.

## Report

The report for this project can be found in the `src/Report.ipynb` file.

To view the report, you need to have Jupyter Notebook installed.

```bash
jupyter notebook src/Report.ipynb
```

## Tensorboard

To visualize the training process, you can use Tensorboard.

```bash
tensorboard --logdir=runs
```
