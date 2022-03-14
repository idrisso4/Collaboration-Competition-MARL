# Collaboration-Competition-MARL

================================================================================


This project is part of Udacity Deep Reinforcement Learning Nanodegree, which is a four-month course that I am enrolled in.
The purpose of this project is to train a MADDPG agent for Unity ML-Agents Tennis environment.


## Environment

For this project, we will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.


![Environment](tennis.png)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


## Install


### Clone the repo

```
git clone https://github.com/idrisso4/Collaboration-Competition-MARL

cd Collaboration-Competition-MARL
```

### Create conda Environment

```
# Create the conda environment
conda create -n deeprlnd python=3.6 numpy=1.13.3 scipy

# Activate the new environment
source activate deeprlnd

# Install dependencies
conda install pytorch torchvision -c pytorch
pip install matplotlib
pip install unityagents==0.4.0
```

### Download the Unity Environment

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Config

Change the parameters in the config.yaml file
(you must change this variable ENVIRONMENT to the location of your downloaded environment)

## Train the Agent:

```
python main.py --train
```

## Evaluate the agent:

```
python main.py --eval
```
