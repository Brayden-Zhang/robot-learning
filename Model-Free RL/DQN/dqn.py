"""
Notes: https://braydenzhang.com/Notes/Reinforcement-Learning/Q-Learning 

Paper Link: https://arxiv.org/abs/1312.5602

Implementation referenced from https://github.com/tanishqkumar/beyond-nanogpt/blob/main/rl/fundamentals/train_dqn.py 

This code implements a Deep Q-Network to solve the CartPole-v1 environment. 

The environment has 4 state variables and 2 actions. 
1. Cart Position
2. Cart Velocity
3. Pole Angle
4. Pole Angular Velocity

Reward: +1 for every step taken, up to 500 steps.
The goal is to balance the pole on the cart for as long as possible.

Algorithm:
1. Initialize replay memory D to capacity N
2. Initialize action-value function Q with random weights
3. For each episode:
    a. Initialize state s
    b. For each step in the episode:
        i. Select action a using epsilon-greedy policy based on Q
        ii. Execute action a and observe reward r and next state s'
        iii. Store transition (s, a, r, s') in replay memory D
        iv. Sample random minibatch from D
        v. Set y = r + γ * max_a' Q(s', a'; θ) for terminal states
        vi. Perform a gradient descent step on (y - Q(s, a; θ))^2 with respect to θ
        vii. Update state s = s'
4. Repeat until convergence
"""
import gym 
import torch
import torch.nn as nn
from tqdm import tqdm 
import argparse
import random
import numpy as np
import os
from gym.wrappers.record_video import RecordVideo

# --- Robust observation extractor ---
def get_obs(obs):
    if isinstance(obs, dict):
        # Try common keys:
        for key in ['observation', 'state']:
            if key in obs:
                return obs[key]
        # If there is only one value in the dict, use it:
        if len(obs) == 1:
            return list(obs.values())[0]
        raise ValueError(f"Could not extract state from dict observation: keys={obs.keys()}")
    return obs


env = gym.make('CartPole-v1', render_mode="rgb_array")
if not os.path.exists('./video'):
    os.makedirs('./video')
env = RecordVideo(env, './video', episode_trigger=lambda episode_number: True)
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class ReplayBuffer: 
    def __init__(self, max_buffer_sz=1001, state_sz=4):
        """
        Initializes the ReplayBuffer with a maximum size and state size.
        Purpose:

        The replay buffer stores experiences (or transitions) that the agent encounters while interacting with the environment. Each experience consists of:

        state: The agent's current state.
        action: The action the agent took in that state.
        reward: The reward the agent received after taking that action.
        next_state: The state the agent transitioned to after taking the action.
        done: A boolean flag indicating whether the episode ended after taking the action.
        """
        self.max_buffer_sz = max_buffer_sz
        
        # Initialize buffers to store states, actions, rewards, next states, and done flags
        self.states = torch.zeros(max_buffer_sz, state_sz) 
        self.actions = torch.zeros(max_buffer_sz)
        self.rewards = torch.zeros(max_buffer_sz)
        self.next_states = torch.zeros(max_buffer_sz, state_sz)
        self.done = torch.zeros(max_buffer_sz)
        
        # Initialize current pointer and size of the buffer
        self.curr_ptr = 0
        self.size = 0    

    def push(self, sartd): 
        """
        Adds a batch of experiences to the replay buffer.

        Args:
            sartd (tuple): A 5-tuple of tensors (states, actions, rewards, next_states, dones).
        """
        bs = sartd[0].shape[0] # Get batch size
        buffers = [self.states, self.actions, self.rewards, self.next_states, self.done]
        for idx, data in enumerate(sartd):     
            buff = buffers[idx]
            space_left = self.max_buffer_sz - self.curr_ptr
            if bs > space_left:
                buff[self.curr_ptr:].copy_(data[:space_left])
                buff[:bs-space_left].copy_(data[space_left:])
            else:
                buff[self.curr_ptr:self.curr_ptr + bs].copy_(data)

        # Update the current pointer and size of the buffer, wrapping around if necessary
        self.curr_ptr = (self.curr_ptr + bs) % self.max_buffer_sz
        self.size = min(self.size + bs, self.max_buffer_sz)
        
        return

    def get_batch(self, bs=64): 
        """
        Samples a random batch of experiences from the replay buffer.

        Args:
            bs (int): The batch size.

        Returns:
            tuple: A tuple of tensors (states, actions, rewards, next_states, dones).
        """
        if self.size < bs: 
            bs = self.size
        
        indices = torch.randint(0, self.size, (bs,)) # Randomly sample indices from the buffer
        states = self.states[indices] 
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.done[indices]
        
        return (states, actions, rewards, next_states, dones)


def loss_function(sartd, policy_net, target_net, gamma=0.95):
    """
    Computes the loss for the DQN algorithm.

    Args:
        sartd (tuple): A tuple of tensors (states, actions, rewards, next_states, dones).
        policy_net (nn.Module): The current policy network.
        target_net (nn.Module): The target network.
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: The computed loss.
    """
    states, actions, rewards, next_states, dones = sartd
    
    # Compute Q-values for the current states
    q_values = policy_net(states).gather(1, actions.long().unsqueeze(1)).squeeze(1)
    
    # Compute Q-values for the next states using the target network
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
    
    # Compute the expected Q-values
    expected_q_values = rewards + (gamma * next_q_values * (1 - dones))
    
    # Compute the loss
    loss = nn.functional.mse_loss(q_values, expected_q_values)
    
    return loss


def train(epochs = 4000, max_buffer_size = 10000, lr = 1e-4, epsilon_final = 0.01, max_rollout_len = 200, num_updates_per_step = 5, train_batch_size = 512, rollout_batch_size = 128, reset_target = 100, verbose = False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = QNet(input_dim=4, output_dim=2).to(device)
    target_net = QNet(input_dim=4, output_dim=2).to(device)

    buffer = ReplayBuffer(max_buffer_sz=max_buffer_size)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    optimizer.zero_grad()

    epsilon=1.0
    epsilon_decay = (epsilon - epsilon_final) / epochs
    n_states = 4

    episode_length_moving_average = 0

    for step in tqdm(range(epochs)):

        rollout_batch_states = torch.zeros(rollout_batch_size, n_states).to(device)
        rollout_batch_actions = torch.zeros(rollout_batch_size).to(device)
        rollout_batch_rewards = torch.zeros(rollout_batch_size).to(device)
        rollout_batch_next_states = torch.zeros(rollout_batch_size, n_states).to(device)
        rollout_batch_dones = torch.zeros(rollout_batch_size).to(device)

        total_experiences = 0

        for _ in range(rollout_batch_size):
            raw_state, _ = env.reset()
            state = get_obs(raw_state)
            done = False
            episode_length = 0

            if state is None or len(state) != 4:
                raise ValueError("Environment state is not initialized correctly. Expected length 4.")

            while not done and episode_length < max_rollout_len:
                episode_length += 1
                rollout_batch_states[_] = torch.tensor(state, dtype=torch.float32).to(device)

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_net(torch.tensor(state, dtype=torch.float32).to(device)).argmax().item()

                rollout_batch_actions[_] = action
                raw_next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = get_obs(raw_next_state)
                done = terminated or truncated
                rollout_batch_rewards[_] = reward
                rollout_batch_next_states[_] = torch.tensor(next_state, dtype=torch.float32).to(device)
                rollout_batch_dones[_] = float(done)

                # env.render()

                state = next_state

            total_experiences += episode_length
 
        buffer.push((rollout_batch_states, rollout_batch_actions, rollout_batch_rewards, 
                rollout_batch_next_states, rollout_batch_dones))


        for i in range(num_updates_per_step): 
            batch = buffer.get_batch(train_batch_size)
     
            loss = loss_function(batch, policy_net, target_net)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epsilon = max(epsilon_final, epsilon - epsilon_decay)

        if step % 200 == 0: 
            # Reward = (moving avg episode len) because we want to keep the pole upright for as long as possible!
            print(f'[{step}/{epochs}]: Loss {loss.item()}, Reward: {round(episode_length_moving_average, 3)}')

        if step % reset_target == 0: 
            target_net.load_state_dict(policy_net.state_dict())

    # Close the environment to finalize video saving
    env.close()

# same structure hypers/args as most files in this project 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN on CartPole')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--buffer-size', type=int, default=2000, help='Size of replay buffer')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epsilon-final', type=float, default=0.01, help='Final exploration rate')
    parser.add_argument('--max-rollout-len', type=int, default=200, help='Maximum rollout length')
    parser.add_argument('--updates-per-step', type=int, default=5, help='Number of updates per step')
    parser.add_argument('--train-batch-size', type=int, default=512, help='Training batch size')
    parser.add_argument('--rollout-batch-size', type=int, default=128, help='Rollout batch size')
    parser.add_argument('--reset-target', type=int, default=100, help='Steps between target network updates')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        max_buffer_size=args.buffer_size,
        lr=args.lr,
        epsilon_final=args.epsilon_final,
        max_rollout_len=args.max_rollout_len,
        num_updates_per_step=args.updates_per_step,
        train_batch_size=args.train_batch_size,
        rollout_batch_size=args.rollout_batch_size,
        reset_target=args.reset_target,
        verbose=args.verbose
    )
