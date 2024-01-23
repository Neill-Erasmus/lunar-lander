from torch.nn import functional as F
from torch import nn
import torch

class NeuralNetwork(nn.Module):
    """
    The NeuralNetwork class defines a simple feedforward neural network architecture
    for a given reinforcement learning task.

    Attributes:
        state_size (int): The size of the input state space.
        action_size (int): The size of the output action space.
        seed (int): A seed for random number generation. Default is 42.

    Methods:
        __init__(state_size, action_size, seed): Constructor method for initializing
            the NeuralNetwork instance with the specified state size, action size, and seed.
        forward(state): Defines the forward pass of the neural network.

    Example:
        # Create an instance of the NeuralNetwork class
        neural_net = NeuralNetwork(state_size=8, action_size=4)
    """

    def __init__(self, state_size : int, action_size : int, seed : int = 42) -> None:
        """
        Initializes the NeuralNetwork instance with the specified state size, action size, and seed.

        Parameters:
            state_size (int): The size of the input state space.
            action_size (int): The size of the output action space.
            seed (int, optional): A seed for random number generation. Default is 42.

        Example:
            # Create an instance of the NeuralNetwork class with specific state and action sizes
            neural_net = NeuralNetwork(state_size=8, action_size=4)
        """

        super(NeuralNetwork, self).__init__()
        self.state_size : int= state_size
        self.seed : torch.Generator = torch.manual_seed(seed)
        self.fc1  : nn.Linear       = nn.Linear(in_features=state_size, out_features=64)
        self.fc2  : nn.Linear       = nn.Linear(in_features=self.fc1.out_features, out_features=64)
        self.fc3  : nn.Linear       = nn.Linear(in_features=self.fc2.out_features, out_features=action_size)

    def forward(self, state):
        """
        Defines the forward pass of the neural network.

        Parameters:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output tensor representing the Q-values for each action.

        Example:
            # Forward pass with a given state tensor
            output = neural_net.forward(torch.tensor([1.0, 2.0, 3.0]))
        """

        x : torch.Tensor = self.fc1(state.view(-1, self.state_size))
        x                = F.relu(input=x)
        x                = self.fc2(x)
        x                = F.relu(input=x)
        return self.fc3(x)