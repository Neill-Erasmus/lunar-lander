from environment import Environment
from agent import Agent
from collections import deque
import numpy
import torch

env   : Environment = Environment(env='LunarLander-v2')
agent : Agent       = Agent(state_size=env.state_size, action_size=env.number_actions)

number_episodes               : int   = 2500
maximum_timesteps_per_episode : int   = 1000
epsilon_starting_value        : float = 1.
episilon_ending_value         : float = 0.01
episilon_decay_value          : float = 0.995
epsilon                       : float = epsilon_starting_value
scores_on_100_episodes        : deque = deque(maxlen = 100)

def train(epsilon : float) -> None:
    """
    Trains the reinforcement learning agent using the specified exploration-exploitation strategy.

    Args:
    - epsilon (float): The initial exploration rate (epsilon-greedy strategy).

    Returns:
    - None

    The function runs training episodes, updating the agent's Q-network and monitoring performance.
    Training stops when the environment is considered solved or the maximum number of episodes is reached.

    During training, the function prints the episode number and the average score over the last 100 episodes.
    If the average score surpasses 200, the training is considered successful, and the agent's model is saved.

    Args:
    - epsilon (float): The initial exploration rate (epsilon-greedy strategy).

    Returns:
    - None

    Example:
    train(epsilon=1.0)
    """

    for episodes in range(1, number_episodes + 1):
        state, _ = env.reset()
        score = 0
        for _ in range(0, maximum_timesteps_per_episode):
            action = agent.action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_on_100_episodes.append(score)
        epsilon = max(episilon_ending_value, episilon_decay_value * epsilon)
        print(f'\rEpisode: {episodes}\tAverage Score: {numpy.mean(scores_on_100_episodes):.2f}',end='')
        if (episodes % 100 == 0):
            print(f'\rEpisode: {episodes}\tAverage Score: {numpy.mean(scores_on_100_episodes):.2f}')
        if numpy.mean(scores_on_100_episodes) >= 200.: #type: ignore
            print(f'\nEnvironment Solved in {episodes:d} episodes!\tAverage Score: {numpy.mean(scores_on_100_episodes):.2f}')
            torch.save(agent.local_qnet.state_dict(), 'model.pth')
            break

def main() -> None:
    """
    Main entry point for training a reinforcement learning agent.

    This function initiates the training process by calling the train function with a specified exploration rate.

    Args:
    - None

    Returns:
    - None

    Example:
    main()
    """

    train(epsilon=epsilon)

if __name__ == "__main__":
    main()