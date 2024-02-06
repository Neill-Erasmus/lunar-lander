# Lunar Lander

https://github.com/Neill-Erasmus/lunar-lander/assets/141222943/5d09c272-d3e8-435e-92a7-bd2477fd629e

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

## The Architecture of the Neural Network

The NeuralNetwork class defines a simple feedforward neural network architecture for reinforcement learning tasks. The architecture consists of three fully connected layers (or linear layers) with rectified linear unit (ReLU) activation functions between them. The input to the network is the state vector, and the output is the Q-values for each action.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
