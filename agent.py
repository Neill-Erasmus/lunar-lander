from neural_network import NeuralNetwork
from replay_memory import ReplayMemory
from hyperparameters import HyperParameters
from torch import optim
from torch.nn import functional as F
from torch import nn
import torch
import random
import numpy

parameters = HyperParameters()

class Agent:
    """
    Represents a reinforcement learning agent using Deep Q-Networks.

    Args:
    - state_size (int): The size of the state space.
    - action_size (int): The number of possible actions.

    Attributes:
    - device (torch.device): The device used for computation (GPU if available, else CPU).
    - state_size (int): The size of the state space.
    - action_size (int): The number of possible actions.
    - local_qnet (NeuralNetwork): The local Q-network for estimating Q-values.
    - target_qnet (NeuralNetwork): The target Q-network for stable learning.
    - optimizer (optim.Adam): The Adam optimizer for updating the local Q-network.
    - memory (ReplayMemory): Replay memory for storing experiences.
    - time_step (int): Counter for tracking time steps.

    Methods:
    - step(state, action, reward, next_state, done): Records a transition in the replay memory and updates the Q-network.
    - action(state, epsilon): Selects an action using an epsilon-greedy strategy.
    - learn(experiences, gamma): Performs a Q-learning update using a batch of experiences.
    - soft_update(local_model, target_model, interpolation_parameter): Updates target model parameters with a soft update.
    """

    def __init__(self, state_size : int, action_size : int) -> None:
        """
        Initializes the Agent with the given state and action sizes.
        """

        self.device      : torch.device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.state_size  : int           = state_size
        self.action_size : int           = action_size
        self.local_qnet  : NeuralNetwork = NeuralNetwork(state_size, action_size).to(self.device)
        self.target_qnet : NeuralNetwork = NeuralNetwork(state_size, action_size).to(self.device)
        self.optimizer   : optim.Adam    = optim.Adam(self.local_qnet.parameters(), lr=parameters.learning_rate)
        self.memory      : ReplayMemory  = ReplayMemory(capacity=parameters.replay_buffer_size)
        self.time_step   : int           = 0

    def step(self, state : numpy.ndarray, action : int, reward : float, next_state : numpy.ndarray, done : bool) -> None:
        """
        Records a transition in the replay memory and updates the Q-network.

        Args:
        - state (numpy.ndarray): The current state.
        - action (int): The taken action.
        - reward (float): The received reward.
        - next_state (numpy.ndarray): The next state.
        - done (bool): Indicates whether the episode is done.
        """

        self.memory.push((state, action, reward, next_state, done))
        self.time_step = (self.time_step + 1) % 4
        if self.time_step == 0:
            if len(self.memory.memory) > parameters.minibatch_size:
                experiences = self.memory.sample(parameters.minibatch_size)
                self.learn(experiences, parameters.gamma)

    def action(self, state : numpy.ndarray, epsilon : float = 0.) -> int:
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
        - state (numpy.ndarray): The current state.
        - epsilon (float): Exploration-exploitation trade-off parameter.

        Returns:
        - int: The selected action.
        """

        state_tensor : torch.Tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnet.eval()
        with torch.no_grad():
            action_values = self.local_qnet(state_tensor)
        self.local_qnet.train()
        if random.random() > epsilon:
            return int(numpy.argmax(action_values.cpu().data.numpy()))
        else:
            return int(random.choice(numpy.arange(self.action_size)))

    def learn(self, experiences : tuple, gamma : float) -> None:
        """
        Performs a Q-learning update using a batch of experiences.

        Args:
        - experiences (tuple): A tuple of (states, next_states, actions, rewards, dones).
        - gamma (float): The discount factor for future rewards.
        """

        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_qnet(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * next_q_targets * (1-dones))
        q_expected = self.local_qnet(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnet, self.target_qnet, parameters.interpolation_parameter)

    def soft_update(self, local_model : nn.Module, target_model : nn.Module, interpolation_parameter : float) -> None:
        """
        Updates target model parameters with a soft update.

        Args:
        - local_model (nn.Module): The source model.
        - target_model (nn.Module): The target model.
        - interpolation_parameter (float): The interpolation parameter.
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)