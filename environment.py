import gym

class Environment():
    """
    The Environment class provides a simple interface for interacting with OpenAI Gym 
    environments. It provides information about the state shape, state size, and the 
    number of available actions in the environment.

    Attributes:
        env (gym.Env): An instance of the 'LunarLander-v2' environment created using OpenAI Gym.
        state_shape (tuple): The shape of the state space in the environment.
        state_size (int): The size of the state space in the environment.
        number_actions (int): The number of available actions in the environment.

    Methods:
        __init__(): Constructor method for initializing the Environment instance.

    Example:
        # Create an instance of the Environment class
        env_instance = Environment()
    """

    def __init__(self, env : str) -> None:
        """
        Initializes the Environment instance with the specified OpenAI Gym environment and
        extracts information about the state shape, state size, and number of actions.

        Parameters:
            env (str, optional): The ID of the OpenAI Gym environment to initialize. 
                                Defaults to 'LunarLander-v2'.

        Example:
            # Create an instance of the Environment class with a custom environment ID
            env_instance = Environment(env='LunarLander-v2')
        """

        self.env            : gym.Env                = gym.make(id=env)
        self.state_shape    : tuple[int, ...] | None = self.env.observation_space.shape
        self.state_size     : int                    = self.state_shape[0] #type: ignore
        self.number_actions : int                    = self.env.action_space.n #type: ignore
        print(f'State Shape: {self.state_shape}\nState Size: {self.state_size}\nNumber of Actions: {self.number_actions}')

    def reset(self) -> tuple:
        """
        Resets the environment to its initial state and returns the initial observation.

        Returns:
            observation: The initial observation/state of the environment.

        Example:
            # Reset the environment and get the initial observation
            initial_observation = env_instance.reset()
        """

        return self.env.reset()

    def step(self, action) -> tuple:
        """
        Takes an action in the environment and returns the next observation, reward, and done flag.

        Parameters:
            action: The action to be taken in the environment.

        Returns:
            observation: The next observation/state of the environment.
            reward: The reward received from the environment.
            done: A flag indicating whether the episode is done.

        Example:
            # Take an action in the environment and get the next observation, reward, and done flag
            next_observation, reward, done = env_instance.step(selected_action)
        """

        return self.env.step(action)