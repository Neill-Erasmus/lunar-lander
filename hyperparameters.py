class HyperParameters():
    """
    The HyperParameters class encapsulates the hyperparameters used in a reinforcement learning model.

    Attributes:
        learning_rate (float): The learning rate for the model's optimizer. Default is 5e-4.
        minibatch_size (int): The size of the minibatch used in training. Default is 100.
        gamma (float): The discount factor for future rewards. Default is 0.99.
        replay_buffer_size (int): The size of the replay buffer for experience replay. Default is 1e5.
        interpolation_parameter (float): A parameter used for interpolation. Default is 1e-3.

    Methods:
        __init__(): Constructor method for initializing the HyperParameters instance.

    Example:
        # Create an instance of the HyperParameters class with default values
        hyperparams = HyperParameters()
    """

    def __init__(self, learning_rate           : float = 5e-4,
                       minibatch_size          : int   = 100,
                       gamma                   : float = 0.99,
                       replay_buffer_size      : int   = int(1e5),
                       interpolation_parameter : float = 1e-3) -> None:
        """
        Initializes the HyperParameters instance with default values for learning rate,
        minibatch size, gamma, replay buffer size, and interpolation parameter.

        Example:
            # Create an instance of the HyperParameters class with default values
            hyperparams = HyperParameters()
        """

        self.learning_rate           = learning_rate
        self.minibatch_size          = minibatch_size
        self.gamma                   = gamma
        self.replay_buffer_size      = replay_buffer_size
        self.interpolation_parameter = interpolation_parameter