import time
import os, sys


#only if using interpreter
# os.chdir('/Users/rsonker001/Documents/Personal/franka_rl_control')
# os.chdir('/home/i53/student/rohit_sonker/franka_rl_control')
# os.chdir('/home/rohit/Documents/franka_rl_control')
# sys.path.append(".")

import numpy as np


from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from wrappers import NormalizeActionWrapper

from stable_baselines3.common.env_checker import check_env


from learning_table_tennis_from_scratch.hysr_one_ball import (
    HysrOneBall,
    HysrOneBallConfig,
)

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

hysr = HysrOneBall_single_robot(hysr_config, reward_function, logs = True)

obs = hysr.reset()

print("example obs = ", obs)
# print("obs shape = ", obs.shape)

check_env(hysr)