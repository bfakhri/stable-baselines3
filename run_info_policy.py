import os

import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py



# Create Atari environment
env_id = "PongNoFrameskip-v4"
n_envs = 4  # Number of environments to run in parallel
env = make_atari_env(env_id, n_envs=n_envs, seed=0)
env = VecFrameStack(env, n_stack=4)

# Initialize the A2C model with the custom policy
model = A2C(ActorCriticInformationCnnPolicy, env, verbose=1)

# Train the agent
model.learn(total_timesteps=20_000)

# Save the trained model
model.save("a2c_atari_custom_policy")

# Optional: Load and test the trained model
# loaded_model = A2C.load("a2c_atari_custom_policy")

# obs = env.reset()
# for i in range(1000):
#     action, _states = loaded_model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
#     if dones.any():
#         obs = env.reset()

# env.close()
