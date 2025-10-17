import gym
import torch
from dqn_agent import DQNAgent
import os

env = gym.make('CartPole-v1', render_mode=None)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
episodes = 500

for e in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.replay()
    print(f"Episode {e+1}/{episodes} - Reward: {total_reward} - Epsilon: {agent.epsilon:.2f}")

# Save model
os.makedirs("../models", exist_ok=True)
agent.save("../models/dqn_model.pth")
print("✅ Training completed and model saved!")
