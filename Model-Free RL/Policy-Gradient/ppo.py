import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # Policy head (mean actions for continuous action space)
        self.policy = nn.Linear(64, act_dim)
        # Value function head
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

# PPO Agent
class PPOAgent:
    def __init__(self, env, clip_param=0.2, gamma=0.99, lam=0.95, 
                 policy_lr=3e-4, epochs=10, batch_size=64):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = ActorCritic(self.obs_dim, self.act_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=policy_lr)
        self.std = nn.Parameter(torch.ones(self.act_dim) * 0.5)

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            mu, _ = self.model(obs)
        dist = torch.distributions.Normal(mu, self.std.exp())
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, trajectories):
        obs = torch.FloatTensor(np.vstack(trajectories['obs'])).to(device)
        actions = torch.FloatTensor(np.vstack(trajectories['acts'])).to(device)
        old_log_probs = torch.FloatTensor(np.vstack(trajectories['log_probs'])).to(device)
        returns = torch.FloatTensor(np.vstack(trajectories['returns'])).to(device)
        advantages = torch.FloatTensor(np.vstack(trajectories['advs'])).to(device)

        for _ in range(self.epochs):
            idxs = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                batch_obs = obs[batch_idx]
                batch_acts = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advs = advantages[batch_idx]

                mu, values = self.model(batch_obs)
                dist = torch.distributions.Normal(mu, self.std.exp())
                new_log_probs = dist.log_prob(batch_acts).sum(-1, keepdim=True)

                # PPO Objective
                ratio = (new_log_probs - batch_old_log_probs).exp()
                clip_adv = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advs
                loss_pi = -(torch.min(ratio * batch_advs, clip_adv)).mean()

                # Value loss
                loss_v = ((batch_returns - values) ** 2).mean()

                # Total loss
                loss = loss_pi + 0.5 * loss_v

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, total_timesteps):
        obs = self.env.reset()
        trajectories = {'obs': [], 'acts': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}
        ep_returns = []
        ep_len = 0
        total_steps = 0

        while total_steps < total_timesteps:
            obs_tensor = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                mu, value = self.model(obs_tensor)
            dist = torch.distributions.Normal(mu, self.std.exp())
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            next_obs, reward, done, _ = self.env.step(action.cpu().numpy())

            trajectories['obs'].append(obs)
            trajectories['acts'].append(action.cpu().numpy())
            trajectories['log_probs'].append(log_prob.cpu().numpy())
            trajectories['rewards'].append(reward)
            trajectories['dones'].append(done)
            trajectories['values'].append(value.item())

            obs = next_obs
            ep_len += 1
            total_steps += 1

            if done:
                obs = self.env.reset()
                ep_returns.append(sum(trajectories['rewards'][-ep_len:]))
                ep_len = 0

            if total_steps % 2048 == 0:
                with torch.no_grad():
                    next_value = self.model(torch.FloatTensor(obs).to(device))[1].item()
                advantages, returns = self.compute_gae(
                    trajectories['rewards'], trajectories['values'], trajectories['dones'], next_value
                )
                trajectories['advs'] = advantages
                trajectories['returns'] = returns
                self.update(trajectories)
                trajectories = {'obs': [], 'acts': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}
                print(f"Step: {total_steps}, Mean Return: {np.mean(ep_returns[-10:])}")

# Instantiate environment and train
env = gym.make("Pendulum-v1")
agent = PPOAgent(env)
agent.train(total_timesteps=10000000)
