#!/usr/bin/env python3
"""
Robot Learning Simulation Environment

This module provides a simulation environment for reinforcement learning
with robots. It includes:
- A configurable robot class with realistic dynamics
- An environment with obstacles and goals
- Visualization tools
- Integration with common RL frameworks
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import gym
from gym import spaces
import time
from IPython.display import clear_output
import random

class RobotArm:
    """A simple 2D robot arm with realistic physics."""
    
    def __init__(self, num_joints=2, link_lengths=None, max_torques=None):
        """
        Initialize a robot arm.
        
        Args:
            num_joints: Number of joints in the robot arm
            link_lengths: List of lengths for each link
            max_torques: Maximum torque that can be applied at each joint
        """
        self.num_joints = num_joints
        self.link_lengths = link_lengths if link_lengths else [1.0] * num_joints
        self.max_torques = max_torques if max_torques else [2.0] * num_joints
        
        # State variables
        self.joint_angles = np.zeros(num_joints)
        self.joint_velocities = np.zeros(num_joints)
        
        # Physical parameters
        self.dt = 0.05  # Time step for simulation (seconds)
        self.damping = 0.1  # Damping coefficient
        self.inertia = [0.1 * length**2 for length in self.link_lengths]  # Moment of inertia
        
        # Keep track of the end effector position
        self.end_effector = self._compute_end_effector()
    
    def _compute_forward_kinematics(self):
        """Compute the positions of all joints including the end effector."""
        points = [(0, 0)]  # Start at origin
        theta = 0
        
        for i in range(self.num_joints):
            theta += self.joint_angles[i]
            x = points[-1][0] + self.link_lengths[i] * np.cos(theta)
            y = points[-1][1] + self.link_lengths[i] * np.sin(theta)
            points.append((x, y))
        
        return np.array(points)
    
    def _compute_end_effector(self):
        """Compute just the end effector position."""
        points = self._compute_forward_kinematics()
        return points[-1]
    
    def apply_torques(self, torques):
        """
        Apply the given torques to the joints and update the dynamics.
        
        Args:
            torques: List of torques to apply to each joint
        """
        # Clip torques to the allowed range
        torques = np.clip(torques, -np.array(self.max_torques), np.array(self.max_torques))
        
        # Apply simple physics model
        for i in range(self.num_joints):
            # Acceleration = torque / inertia - damping * velocity
            acceleration = (torques[i] / self.inertia[i]) - (self.damping * self.joint_velocities[i])
            
            # Update velocity and position using Euler integration
            self.joint_velocities[i] += acceleration * self.dt
            self.joint_angles[i] += self.joint_velocities[i] * self.dt
            
            # Normalize angles to -pi to pi
            self.joint_angles[i] = ((self.joint_angles[i] + np.pi) % (2 * np.pi)) - np.pi
        
        # Update end effector position
        self.end_effector = self._compute_end_effector()
        
    def get_state(self):
        """Return the current state of the robot."""
        return {
            'joint_angles': self.joint_angles.copy(),
            'joint_velocities': self.joint_velocities.copy(),
            'end_effector': self.end_effector.copy()
        }


class Obstacle:
    """Represents an obstacle in the environment."""
    
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def check_collision(self, point):
        """Check if a point collides with this obstacle."""
        return (self.x <= point[0] <= self.x + self.width and
                self.y <= point[1] <= self.y + self.height)


class RobotEnv(gym.Env):
    """
    A reinforcement learning environment for a robot arm.
    
    Task: Move the end effector to a goal position while avoiding obstacles.
    """
    
    def __init__(self, num_joints=2, max_steps=100):
        super().__init__()
        
        self.robot = RobotArm(num_joints=num_joints)
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(num_joints,),
            dtype=np.float32
        )
        
        # Observation: joint angles, joint velocities, goal position, obstacles
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(num_joints * 2 + 2,),  # angles, velocities, goal_x, goal_y
            dtype=np.float32
        )
        
        # Environment setup
        self.obstacles = []
        self.goal_position = None
        self.goal_radius = 0.2
        self.workspace_bounds = (-3, 3, -3, 3)  # (min_x, max_x, min_y, max_y)
        
        # Initialize the environment
        self.reset()
    
    def _generate_random_obstacles(self, num_obstacles=3):
        """Generate random obstacles that don't overlap with the start or goal."""
        obstacles = []
        for _ in range(num_obstacles):
            while True:
                x = np.random.uniform(self.workspace_bounds[0], self.workspace_bounds[1] - 0.5)
                y = np.random.uniform(self.workspace_bounds[2], self.workspace_bounds[3] - 0.5)
                width = np.random.uniform(0.3, 0.7)
                height = np.random.uniform(0.3, 0.7)
                
                # Check if obstacle overlaps with goal
                if np.linalg.norm(np.array([x + width/2, y + height/2]) - self.goal_position) < (self.goal_radius + max(width, height)/2):
                    continue
                
                # Check if obstacle overlaps with robot starting position
                # (simplified check just with end effector)
                if np.linalg.norm(np.array([x + width/2, y + height/2]) - self.robot.end_effector) < max(width, height):
                    continue
                
                obstacles.append(Obstacle(x, y, width, height))
                break
        
        return obstacles
    
    def _generate_goal(self):
        """Generate a random goal position within the workspace."""
        while True:
            x = np.random.uniform(self.workspace_bounds[0] + 0.5, self.workspace_bounds[1] - 0.5)
            y = np.random.uniform(self.workspace_bounds[2] + 0.5, self.workspace_bounds[3] - 0.5)
            
            # Make sure goal is reachable by checking if it's within the maximum reach of the arm
            max_reach = sum(self.robot.link_lengths)
            if np.sqrt(x**2 + y**2) <= max_reach * 0.9:  # 90% of max reach to ensure it's comfortably reachable
                return np.array([x, y])
    
    def reset(self):
        """Reset the environment to a new initial state."""
        # Reset robot
        self.robot.joint_angles = np.random.uniform(-np.pi/4, np.pi/4, self.robot.num_joints)
        self.robot.joint_velocities = np.zeros(self.robot.num_joints)
        self.robot.end_effector = self.robot._compute_end_effector()
        
        # Generate new goal
        self.goal_position = self._generate_goal()
        
        # Generate new obstacles
        self.obstacles = self._generate_random_obstacles()
        
        self.current_step = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """Return the current observation."""
        return np.concatenate([
            self.robot.joint_angles,
            self.robot.joint_velocities,
            self.goal_position
        ])
    
    def _compute_reward(self, action):
        """
        Compute the reward for the current state and action.
        
        Reward components:
        1. Distance to goal (negative)
        2. Reaching the goal (positive)
        3. Collision with obstacles (negative)
        4. Action magnitude (small negative to encourage efficient movement)
        """
        # Compute distance to goal
        distance_to_goal = np.linalg.norm(self.robot.end_effector - self.goal_position)
        
        # Base reward is negative distance to goal
        reward = -distance_to_goal
        
        # Check if goal reached
        if distance_to_goal < self.goal_radius:
            reward += 100.0
        
        # Check for collisions with obstacles
        for obstacle in self.obstacles:
            if obstacle.check_collision(self.robot.end_effector):
                reward -= 50.0
                break
        
        # Small penalty for large actions
        action_penalty = -0.01 * np.sum(np.square(action))
        reward += action_penalty
        
        return reward
    
    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Args:
            action: The torques to apply to the robot joints, scaled from -1 to 1
        
        Returns:
            observation, reward, done, info
        """
        # Scale the action to the actual torque range
        scaled_action = action * np.array(self.robot.max_torques)
        
        # Apply the action to the robot
        self.robot.apply_torques(scaled_action)
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check if done
        done = False
        info = {}
        self.current_step += 1
        
        # Check if goal reached
        distance_to_goal = np.linalg.norm(self.robot.end_effector - self.goal_position)
        if distance_to_goal < self.goal_radius:
            done = True
            info['success'] = True
        
        # Check if out of bounds
        x, y = self.robot.end_effector
        if (x < self.workspace_bounds[0] or x > self.workspace_bounds[1] or
            y < self.workspace_bounds[2] or y > self.workspace_bounds[3]):
            done = True
            reward -= 30.0
            info['out_of_bounds'] = True
        
        # Check if hit an obstacle
        for obstacle in self.obstacles:
            if obstacle.check_collision(self.robot.end_effector):
                info['collision'] = True
                # Don't end the episode, just penalize
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            done = True
            info['timeout'] = True
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """Render the current state of the environment."""
        plt.figure(figsize=(10, 8))
        plt.clf()
        
        # Set axis limits based on workspace bounds
        plt.xlim(self.workspace_bounds[0], self.workspace_bounds[1])
        plt.ylim(self.workspace_bounds[2], self.workspace_bounds[3])
        
        # Draw obstacles
        for obstacle in self.obstacles:
            rect = Rectangle((obstacle.x, obstacle.y), obstacle.width, obstacle.height,
                            color='red', alpha=0.5)
            plt.gca().add_patch(rect)
        
        # Draw goal
        goal_circle = Circle(self.goal_position, self.goal_radius, color='green', alpha=0.5)
        plt.gca().add_patch(goal_circle)
        
        # Draw robot arm
        points = self.robot._compute_forward_kinematics()
        plt.plot(points[:, 0], points[:, 1], 'bo-', linewidth=2, markersize=8)
        
        # Draw end effector
        plt.plot(self.robot.end_effector[0], self.robot.end_effector[1], 'ro', markersize=10)
        
        plt.grid(True)
        plt.title(f'Robot Arm Simulation - Step: {self.current_step}')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        
        if mode == 'human':
            plt.pause(0.01)
            plt.show(block=False)
            clear_output(wait=True)
        elif mode == 'rgb_array':
            # This is more complex and requires converting the matplotlib figure to an array
            # Not fully implemented here for simplicity
            pass


class TrainingUtils:
    """Utilities for training reinforcement learning agents."""
    
    @staticmethod
    def train_random_agent(env, episodes=10, render_every=1):
        """Train a random agent as a baseline."""
        rewards = []
        
        for episode in range(episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                # Take random action
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                step += 1
                
                # Render periodically
                if episode % render_every == 0:
                    env.render()
                    time.sleep(0.01)
            
            rewards.append(episode_reward)
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step}")
        
        return rewards

    @staticmethod
    def simple_policy(observation, env):
        """
        A simple hand-crafted policy that moves toward the goal.
        
        This function demonstrates how to create a basic policy that doesn't
        require learning, but shows the concept of mapping observations to actions.
        """
        # Extract joint angles, velocities, and goal position
        num_joints = env.robot.num_joints
        joint_angles = observation[:num_joints]
        goal_x, goal_y = observation[-2:]
        
        # Current end effector position
        end_x, end_y = env.robot.end_effector
        
        # Direction to goal
        dx = goal_x - end_x
        dy = goal_y - end_y
        
        # For a simple 2-joint robot, we can use a basic inverse kinematics approach
        if num_joints == 2:
            l1, l2 = env.robot.link_lengths
            target_x, target_y = goal_x, goal_y
            
            # Clamp target to max reach if necessary
            max_reach = l1 + l2
            target_dist = np.sqrt(target_x**2 + target_y**2)
            if target_dist > max_reach * 0.99:
                target_x = target_x * max_reach * 0.99 / target_dist
                target_y = target_y * max_reach * 0.99 / target_dist
            
            # Simple inverse kinematics for 2-joint arm
            try:
                # Calculate joint angles using inverse kinematics
                cos_theta2 = (target_x**2 + target_y**2 - l1**2 - l2**2) / (2 * l1 * l2)
                cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  # Prevent numerical errors
                theta2 = np.arccos(cos_theta2)
                # Choose elbow up or down configuration based on target position
                if target_y < 0:
                    theta2 = -theta2
                
                # Calculate first joint angle
                theta1 = np.arctan2(target_y, target_x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
                
                # Calculate desired joint angles
                target_angles = np.array([theta1, theta2])
                
                # Simple proportional control to move towards target angles
                kp = 1.0  # Proportional gain
                action = kp * (target_angles - joint_angles) / env.robot.max_torques
                
                # Clip to action space
                return np.clip(action, -1.0, 1.0)
                
            except:
                # Fallback to a simpler approach if IK fails
                pass
        
        # Fallback approach: Move in the general direction of the goal
        # Scale action based on distance to goal
        action_scale = min(1.0, np.sqrt(dx**2 + dy**2))
        
        # Create a simple action that tries to reach toward the goal
        # This is an extremely simplified approach
        actions = np.zeros(num_joints)
        for i in range(num_joints):
            # Even joints control x-movement, odd joints control y-movement
            # This is a very naive approach and will only work in limited scenarios
            if i % 2 == 0:
                actions[i] = np.sign(dx) * action_scale
            else:
                actions[i] = np.sign(dy) * action_scale
        
        return np.clip(actions, -1.0, 1.0)


# Example usage
if __name__ == "__main__":
    # Create environment
    env = RobotEnv(num_joints=2, max_steps=100)
    
    # Test random agent
    print("Training random agent...")
    random_rewards = TrainingUtils.train_random_agent(env, episodes=5, render_every=1)
    
    print("Testing simple policy...")
    
    # Test simple policy
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        action = TrainingUtils.simple_policy(obs, env)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        time.sleep(0.05)
    
    print(f"Simple policy finished with total reward: {total_reward:.2f}")
    
    # Integration example with an RL algorithm (pseudocode)
    print("""
    # To integrate with PPO or other RL algorithms:
    # 1. Create the environment
    env = RobotEnv(num_joints=2)
    
    # 2. Import your preferred RL library (e.g., Stable Baselines 3)
    # from stable_baselines3 import PPO
    
    # 3. Create and train the agent
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=10000)
    
    # 4. Evaluate the trained agent
    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()
    """)