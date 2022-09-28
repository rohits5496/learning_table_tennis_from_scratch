import time
import os, sys


#only if using interpreter
# os.chdir('/Users/rsonker001/Documents/Personal/franka_rl_control')
# os.chdir('/home/i53/student/rohit_sonker/franka_rl_control')
# os.chdir('/home/rohit/Documents/franka_rl_control')
# sys.path.append(".")

import numpy as np


from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize, VecFrameStack
from wrappers import NormalizeActionWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations


from stable_baselines3.common.env_checker import check_env


from learning_table_tennis_from_scratch.hysr_one_ball import (
    HysrOneBall,
    HysrOneBallConfig,
)

from stable_baselines3 import PPO
import logging
from one_ball_alt import HysrOneBall_single_robot
from learning_table_tennis_from_scratch.rewards import JsonReward

logging.basicConfig(format="hysr_one_ball_swing | %(message)s", level=logging.INFO)
# reward_config_path, hysr_config_path = _configure()
hysr_config_path = "config/hysr_single_robot_traj.json"
reward_config_path = "config/reward_default.json"
print(
"\nusing configuration files:\n- {}\n- {}\n".format(reward_config_path, hysr_config_path)
)

hysr_config = HysrOneBallConfig.from_json(hysr_config_path)
reward_function = JsonReward.get(reward_config_path)
algo_time_step = hysr_config.algo_time_step

env = HysrOneBall_single_robot(hysr_config, reward_function, logs = True)
env = Monitor(env)
env = NormalizeActionWrapper(env)


env = DummyVecEnv([lambda:env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.0)
env = VecFrameStack(env, n_stack = 4)

obs = env.reset()

print("example obs = ", obs)
print("SAMPLES")
print("obs sample = ",env.observation_space.sample())
print("Act sample = ",env.action_space.sample())

check_env(env)

# print("obs shape = ", obs.shape)

# model = PPO('MlpPolicy', env, verbose=1, gamma=0.0, device = 'cpu')

# model.learn(total_timesteps=500*2, log_interval = 1)