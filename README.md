# Q-learning Agent Implementation

This project contains an implementation of a Q-learning agent in Python for reinforcement learning tasks with a visual representation in `pygame` The agent interacts with an environment (like the Frozen Lake environment) and learns to make decisions through trial and error, using a method called Q-learning.

## QAgent Class

The `QAgent` class implements a basic Q-learning algorithm. This agent maintains a Q-table where it stores the expected utility (or quality) of taking a given action in a given state. Over time, the agent learns to choose actions that maximize the expected reward.

### Key Components of the QAgent

- **Q-table Initialization**: The Q-table is initialized with a specified value (default is 0.0), and its size is based on the number of states and actions in the environment.
  
- **Learning (`learn` method)**: The agent updates its Q-table based on the rewards it receives from the environment. The update rule follows the Q-learning formula:

```Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(new_state, a')) - Q(s, a)]```


where:
- `Q(s, a)` is the current Q-value for state `s` and action `a`.
- `alpha` is the learning rate.
- `reward` is the reward received after taking action `a` in state `s`.
- `gamma` is the discount factor, which determines the importance of future rewards.
- `new_state` is the state after taking action `a`.
- `a'` represents possible actions in the new state.

- **Action Selection (`choose_action` method)**: The agent uses an ε-greedy policy to balance exploration and exploitation:
- With probability `epsilon`, the agent chooses a random action (exploration).
- With probability `1 - epsilon`, the agent chooses the action with the highest Q-value for the current state (exploitation).

- **Policy Update (`update_action_policy` method)**: The agent gradually reduces its exploration rate (`epsilon`) after each step, allowing it to focus more on exploiting learned knowledge as training progresses.

- **Saving and Loading**: The agent can save its Q-table to a file and load it later, which is useful for continuing training or testing with a pre-trained model.

### Key Hyperparameters

- **Learning Rate (lr = 0.2)**: Controls how much new information overrides the old information. A higher learning rate means the agent considers new information more heavily.

- **Discount Factor (gamma = 0.9)**: Balances the importance of immediate and future rewards. A discount factor close to 1.0 means future rewards are considered very important, whereas a lower value means the agent focuses more on immediate rewards.

- **Epsilon (epsilon = 0.9)**: The initial probability of choosing a random action (exploration). This encourages the agent to explore the environment before exploiting known information.

- **Epsilon Decrement (eps_decrement = 0.01)**: The amount by which `epsilon` decreases after each episode. This reduces exploration over time as the agent becomes more confident in its learned policy.

- **Epsilon Minimum (eps_min = 0.01)**: The minimum value that `epsilon` can reach, ensuring that the agent continues to explore with a small probability even after extensive training.

## Running the Q-learning Agent

To run the Q-learning agent, specify the desired mode (`train` or `test`) and the type of agent (`q_learning` or `manual`). The agent will interact with the environment and either learn from scratch (training mode) or test a pre-trained model (testing mode).

### Example Command

```python
if __name__ == '__main__':
  agent = 'q_learning'  # Choose between 'q_learning' or 'manual'
  render = True
  mode = 'test'  # 'train' or 'test'

  # Initialize the environment
  env = FrozenLake()

  # Initialize the agent
  agent = QAgent(n_states=env.n_states, n_actions=env.n_actions)

  # Run the main pygame loop
  main_pygame(env, agent, save_path=save_path, render=render, num_episodes=5000, test_mode=(mode == 'test'))
```

This command sets up the environment and agent, and then runs the agent through the specified number of episodes.

## Saving and Loading Models
The agent's state (Q-table) can be saved and loaded to/from a file. This is useful for saving progress during training or for deploying a pre-trained agent in a testing environment.

## Summary
This Q-learning implementation is a simple yet powerful demonstration of reinforcement learning principles. By tuning the hyperparameters and experimenting with different environments, users can explore how the Q-learning algorithm converges to optimal policies.
