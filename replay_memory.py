import random
import numpy
import torch

class ReplayMemory(object):
    """
    The ReplayMemory class represents a replay memory buffer for storing and sampling
    experiences for reinforcement learning.

    Attributes:
        capacity (int): The maximum capacity of the replay memory.
        device (torch.device): The device on which the memory is stored (CPU or CUDA).
        memory (list): List to store experiences.

    Methods:
        __init__(capacity): Constructor method for initializing the ReplayMemory instance.
        push(event): Adds an experience tuple to the replay memory.
        sample(batch_size): Randomly samples a batch of experiences from the replay memory.

    Example:
        # Create an instance of the ReplayMemory class with capacity 1000
        memory = ReplayMemory(capacity=1000)
    """

    def __init__(self, capacity : int) -> None:
        """
        Initializes the ReplayMemory instance with the specified capacity.

        Parameters:
            capacity (int): The maximum capacity of the replay memory.

        Example:
            # Create an instance of the ReplayMemory class with capacity 1000
            memory = ReplayMemory(capacity=1000)
        """

        self.device   : torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.capacity : int          = capacity
        self.memory   : list         = []

    def push(self, event : tuple) -> None:
        """
        Adds an experience tuple to the replay memory.

        Parameters:
            event (tuple): The experience tuple to be added.

        Example:
            # Add an experience tuple to the replay memory
            memory.push((state, action, reward, next_state, done))
        """

        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size : int) -> tuple:
        """
        Randomly samples a batch of experiences from the replay memory.

        Parameters:
            batch_size (int): The number of experiences to be sampled in a batch.

        Returns:
            tuple: A tuple containing PyTorch tensors for states, next_states, actions,
                   rewards, and dones.

        Example:
            # Sample a batch of experiences from the replay memory
            states, next_states, actions, rewards, dones = memory.sample(batch_size=64)
        """

        experiences : list         = random.sample(self.memory, k=batch_size)
        states      : torch.Tensor = torch.from_numpy(numpy.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions     : torch.Tensor = torch.from_numpy(numpy.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards     : torch.Tensor = torch.from_numpy(numpy.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states : torch.Tensor = torch.from_numpy(numpy.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones       : torch.Tensor = torch.from_numpy(numpy.vstack([e[4] for e in experiences if e is not None]).astype(numpy.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones