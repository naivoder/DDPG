# Deep Deterministic Policy Gradients (DDPG)

This repository contains an implementation of the Deep Deterministic Policy Gradients (DDPG) algorithm. The implementation is evaluated on various standard continuous control environments from the Gymnasium library.

## Overview

DDPG is an actor-critic algorithm designed for environments with continuous action spaces. This implementation includes components such as an actor network, a critic network, replay memory, and action noise.

## Setup

### Requirements

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Running the Algorithm

You can run the algorithm on any supported Gymnasium environment. For example:

```bash
python main.py --env 'LunarLanderContinuous-v2'
```

## Acknowledgements

This implementation is based on the DDPG algorithm as described in the paper ["Continuous control with deep reinforcement learning" by Lillicrap et al](https://arxiv.org/abs/1509.02971).
