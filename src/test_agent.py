import gym
import torch
from dqn_agent import DQNAgent

env = gym.make('CartPole-v1', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
agent.load("../models/dqn_model.pth")
agent.epsilon = 0.0  # Disable exploration during testing

for episode in range(5):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    print(f"🏆 Test Episode {episode+1}: Total Reward = {total_reward}")

env.close()
