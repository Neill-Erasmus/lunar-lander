# Lunar Lander

<p align="center">
  <img src="https://github.com/Neill-Erasmus/lunar-lander/assets/141222943/8887d155-1211-4f24-a164-222219af185b" alt="lunar_lander">
</p>

A Deep Q-Network (DQN) has been developed and trained to master the Lunar Lander environment within OpenAI's Gymnasium. This project is dedicated to harnessing reinforcement learning methodologies, empowering an agent to independently navigate and successfully land a lunar module on the moon's surface.

## Deep Q Learning

Deep Q-Learning (DQN) is a powerful algorithm in the field of reinforcement learning, designed to enable an agent to learn optimal strategies for decision-making in environments with discrete action spaces. The "Q" in DQN refers to the Q-function, which represents the expected cumulative reward of taking a particular action in a given state and following the optimal policy thereafter.

### State Representation

The agent interacts with an environment, and at each time step, it observes a state s. 
The state can be a representation of the environment's current configuration or situation.

### Action Selection

Based on the observed state s, the agent selects an action a from its set of possible actions.
The selection is typically guided by a policy that determines the agent's decision-making strategy.

### Environment Interaction

The selected action is applied to the environment, resulting in a transition to a new state s and a reward r. 
The environment's response to the agent's action provides feedback on the quality of the decision made.

### Experience Replay

The agent stores these experiences (s,a,r,s) in a replay memory. Experience replay involves randomly sampling 
from this memory during the learning process. This helps break the temporal correlation between consecutive 
experiences, leading to more stable and efficient learning.

### Q-Network

The core of DQN is a neural network (Q-network) that takes a state s as input and outputs Q-values for all possible actions.
The Q-values represent the expected cumulative reward of taking each action in the given state.

### Target Q-Network

To stabilize the learning process, DQN introduces a target Q-network. This is a separate neural network with the same architecture as the Q-network. The target Q-network is periodically updated with the weights from the Q-network.

### Q-Value Update (Temporal Difference Learning)

The Q-network is trained using a form of temporal difference learning. The loss function is defined as the Mean Squared Error (MSE) between the predicted Q-values and the target Q-values. The target Q-values are calculated as the sum of the immediate reward and the discounted maximum Q-value of the next state.

### Exploration-Exploitation

To balance exploration and exploitation, an epsilon-greedy strategy is often employed. The agent chooses the action with the highest Q-value with probability 1−ϵ and explores new actions with probability ϵ.

### Training Iterations

The agent iteratively interacts with the environment, collects experiences, and updates the Q-network through backpropagation. The training process continues until the Q-network converges to an optimal set of Q-values.

By combining neural networks, experience replay, and target networks, DQN has proven effective in learning complex strategies for a wide range of tasks, including playing video games, robotic control, and navigation in simulated environments.

## Overview of Lunar Lander Environment

<p align="center">
  <img src="https://github.com/Neill-Erasmus/lunar-lander/assets/141222943/8887d155-1211-4f24-a164-222219af185b" alt="lunar_lander">
</p>

The Lunar Lander environment is a part of the Box2D environments collection, providing a simulation of a spacecraft landing on the moon. Here's an overview of its key aspects:

### Action Space:

Discrete(4): The agent can choose from four discrete actions:
0. Do nothing
Fire left orientation engine
Fire main engine
Fire right orientation engine

### Observation Space:

Box([-1.5, -1.5, -5., -5., -3.1415927, -5., -0., -0.], [1.5, 1.5, 5., 5., 3.1415927, 5., 1., 1.], (8,), float32): The environment provides an observation space consisting of an 8-dimensional vector. This includes the coordinates of the lander in x and y, its linear velocities in x and y, its angle, its angular velocity, and two booleans representing whether each leg is in contact with the ground.

### Description:

The Lunar Lander environment presents a classic rocket trajectory optimization problem. It follows Pontryagin’s maximum principle, suggesting it's optimal to either fire the engine at full throttle or turn it off, hence the discrete actions.
Two versions of the environment exist: discrete or continuous. The landing pad is fixed at coordinates (0,0), allowing for the possibility of landing outside it. Fuel is infinite, enabling agents to learn to fly and land effectively on their first attempt.

### Rewards:

Rewards are granted after each step, with the total episode reward being the sum of individual step rewards.
Rewards increase or decrease based on the proximity to the landing pad, the speed of the lander, its tilt angle, and the contact status of its legs.
Additional rewards or penalties are given for engine firing, landing, or crashing, with a successful landing yielding +100 points and a crash incurring a penalty of -100 points.

### Starting State:

The lander begins at the top center of the viewport with a random initial force applied to its center of mass.
The Lunar Lander environment provides a challenging yet rewarding task for reinforcement learning agents to master the intricacies of spacecraft control and landing.

## The Architecture of the Neural Network

The NeuralNetwork class defines a simple feedforward neural network architecture for reinforcement learning tasks. The architecture consists of three fully connected layers (or linear layers) with rectified linear unit (ReLU) activation functions between them. The input to the network is the state vector, and the output is the Q-values for each action.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
