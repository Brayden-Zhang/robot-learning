import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import trange

# Hyperparameters
ENV_NAME = "highway-v0"
NUM_EPISODES = 1000
GAMMA = 0.99
LR = 3e-4
ENTROPY_COEF = 1e-2
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
N_STEPS = 5     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42



config = {
    "policy_frequency": 2,

}


np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- Actor-Critic Neural Network ----
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value

# ---- Rollout Buffer for N-Step Experience Collection ----
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

    def append(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

# ---- Helper: Compute Returns and Advantages ----
def compute_returns_and_advantages(buffer, gamma, next_value, next_done):
    returns = []
    advantages = []
    R = next_value
    for step in reversed(range(len(buffer.rewards))):
        mask = 1.0 - float(buffer.dones[step])
        R = buffer.rewards[step] + gamma * R * mask
        returns.insert(0, R)
        advantages.insert(0, R - buffer.values[step])
    return torch.tensor(returns, dtype=torch.float32, device=DEVICE), \
           torch.tensor(advantages, dtype=torch.float32, device=DEVICE)

# ---- Main Training Loop ----
def train():
    env = gym.make(ENV_NAME, render_mode=None, config=config)
    # env.config['policy_frequency'] = 2  # makes learning easier
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = ActorCritic(obs_dim, n_actions).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode_rewards = deque(maxlen=100)

    obs, _ = env.reset(seed=SEED)
    for episode in trange(NUM_EPISODES):
        buffer = RolloutBuffer()
        ep_reward = 0
        done = False

        while not done:
            # Collect N_STEPS transitions
            for _ in range(N_STEPS):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # Shape: (1, obs_dim)
                logits, value = model(obs_tensor)        # logits: (1, n_actions), value: (1,)
                logits = logits.squeeze(0)               # (n_actions,)
                value = value.squeeze(0)                 # ()
                probs = torch.softmax(logits, dim=-1)    # (n_actions,)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()                   # Scalar tensor
                log_prob = dist.log_prob(action)         # Scalar

                next_obs, reward, terminated, truncated, _ = env.step(action.item())

                done_flag = terminated or truncated
                buffer.append(obs, action.item(), log_prob.item(), reward, done_flag, value.item())

                obs = next_obs
                ep_reward += reward
                if done_flag:
                    break

            # Compute bootstrap value for final state in trajectory
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                _, next_value = model(obs_tensor)
            returns, advantages = compute_returns_and_advantages(
                buffer, GAMMA, next_value.item() * (not done_flag), done_flag
            )

            # Prepare batch tensors
            obs_batch = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=DEVICE)
            action_batch = torch.tensor(buffer.actions, dtype=torch.int64, device=DEVICE)
            log_prob_batch = torch.tensor(buffer.log_probs, dtype=torch.float32, device=DEVICE)
            value_batch = torch.tensor(buffer.values, dtype=torch.float32, device=DEVICE)

            # Forward pass for policy and value
            logits, values = model(obs_batch)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(action_batch)
            entropy = dist.entropy().mean()

            # Losses
            policy_loss = -(advantages.detach() * new_log_probs).mean()
            value_loss = VALUE_LOSS_COEF * (returns.detach() - values).pow(2).mean()
            entropy_loss = -ENTROPY_COEF * entropy
            loss = policy_loss + value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            buffer.clear()

            if done_flag:
                break

        episode_rewards.append(ep_reward)
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode}: Reward={ep_reward:.2f}  Avg(100)={avg_reward:.2f}")

    env.close()

if __name__ == "__main__":
    train()
