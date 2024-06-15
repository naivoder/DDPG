# Deep Deterministic Policy Gradients (DDPG)

This repository contains an implementation of the Deep Deterministic Policy Gradients (DDPG) algorithm. The implementation is evaluated on various standard continuous control environments from the Gymnasium library.

## Overview

DDPG is an actor-critic algorithm designed for environments with continuous action spaces. This implementation includes components such as an actor network, a critic network, replay memory, and action noise.

The paper introduces Deep Deterministic Policy Gradient (DDPG), an actor-critic, model-free algorithm tailored to continuous action domains. Building on the deterministic policy gradient (DPG) framework, DDPG adapts techniques from Deep Q-Network (DQN) to handle high-dimensional, continuous action spaces. The algorithm has been tested successfully on over 20 simulated physics tasks, demonstrating its effectiveness across a variety of complex environments, including challenges like dexterous manipulation and autonomous driving.

### Key Highlights

- **Extension to Continuous Actions:** DDPG extends the ideas from DQN to continuous action spaces, addressing the limitations of DQN which is confined to discrete actions.
- **Actor-Critic Architecture:** The implementation uses an actor-critic framework, where the actor directly maps states to actions and the critic evaluates the action by computing the value function.
- **Stability Enhancements:** Incorporating techniques such as experience replay and target networks from DQN helps stabilize the training with function approximators in continuous action domains.

Utilization of Batch Normalization: DDPG incorporates batch normalization to manage the diverse scale of inputs effectively, especially when dealing with different physical units and varied ranges across environments. This normalization standardizes each input dimension to have zero mean and unit variance within each mini-batch, facilitating faster and more stable training. This technique not only helps in adapting the network to changes in input scales without manual tuning but also enhances the generalization of the model across different tasks.

End-to-End Learning from Pixels: DDPG demonstrates the ability to learn policies directly from raw pixel inputs across various tasks, proving the algorithm's robustness and versatility.

### Summary of Gradient Updates During Learning

#### Critic Loss

The critic loss function is taken with respect to the parameters of the critic network, not the actions directly. The critic network approximates the action-value function \( Q \), and the goal is to minimize the error in this approximation. The critic loss function is defined as:

\[ L(\theta^Q) = \mathbb{E}_{s, a, r, s'} \left[ (r + \gamma Q'(s', \mu'(s'|\theta^{\mu'})) - Q(s, a|\theta^Q))^2 \right] \]

where \( Q' \) and \( \mu' \) are target networks for the critic and actor, respectively.

The gradient of the critic loss with respect to the critic parameters \( \theta^Q \) is given by:

\[ \nabla_{\theta^Q} L = \mathbb{E}_{s, a, r, s'} \left[ (r + \gamma Q'(s', \mu'(s'|\theta^{\mu'})) - Q(s, a|\theta^Q)) \nabla_{\theta^Q} Q(s, a|\theta^Q) \right] \]

#### Actor Loss

The actor's objective is to maximize the expected return, which is achieved by taking actions that maximize the action-value function as estimated by the critic. The actor does not have a traditional loss function like the critic; instead, it is updated using the policy gradient. The gradient of the expected return \( J \) with respect to the actor parameters is given by:

\[ \nabla_{\theta^\mu} J \approx \mathbb{E}_{s} \left[ \nabla_a Q(s, a|\theta^Q) |_{a=\mu(s)} \nabla_{\theta^\mu} \mu(s|\theta^\mu) \right] \]

This gradient shows how the parameters \( \theta^\mu \) of the actor network should be adjusted to improve the policy. Specifically, the actor's parameters are updated to increase the expected action-value \( Q \).

#### Relationship Between Actor and Critic Updates

- **Critic**: The critic evaluates the action-value function \( Q \) by minimizing the loss function with respect to its parameters \( \theta^Q \).
- **Actor**: The actor's parameters \( \theta^\mu \) are updated to maximize the action-value function \( Q \), where \( a = \mu(s) \). This is done by taking the gradient of the action-value function with respect to the action, and then with respect to the actor's parameters.

#### Summary

- The critic's loss function is minimized with respect to the critic's parameters \( \theta^Q \).
- The actor's policy is improved by maximizing the action-value function, and the gradient for the actor is taken with respect to the actor's parameters \( \theta^\mu \).

So, while the critic does evaluate the actions taken, its loss function is ultimately about adjusting its own parameters to better approximate \( Q \). Conversely, the actor updates its parameters to take better actions according to the critic's evaluations.

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

## Environments

_LunarLanderContinuous-v2_
Best Score:

## Acknowledgements

This implementation is based on the DDPG algorithm as described in the paper ["Continuous control with deep reinforcement learning" by Lillicrap et al](https://arxiv.org/abs/1509.02971).
