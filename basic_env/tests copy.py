import gym
import numpy as np
import os
print(os.getcwd())
import logging
import o80

import sys
sys.path.append('/home/rohit/Documents/learning_table_tennis_from_scratch/')

from env import PAMenv
from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv
from learning_table_tennis_from_scratch.hysr_one_ball import (
    HysrOneBall,
    HysrOneBallConfig,
)
from learning_table_tennis_from_scratch.rewards import JsonReward
from learning_table_tennis_from_scratch.jsonconfig import get_json_config


hysr_config_path = 'config/hysr_demos.json'
reward_config_path = 'config/reward_default.json'

hysr_config = HysrOneBallConfig.from_json(hysr_config_path)
reward_function = JsonReward.get(reward_config_path)
algo_time_step = hysr_config.algo_time_step

print("Creating env...")
hysr = HysrOneBall(hysr_config, reward_function)

# print("Observation space:", hysr.observation_space)
# print("Shape:", hysr.observation_space.shape)
# # Discrete(2) means that there is two discrete actions
# print("Action space:", hysr.action_space)

trajectory_index = 49
hysr.set_ball_behavior(index=trajectory_index)

hysr.reset()

frequency_manager = o80.FrequencyManager(1.0 / algo_time_step)

swing_posture = [[18000, 17000], [16800, 19100], [18700, 17300], [18000, 18000]]
swing_pressures = [p for sublist in swing_posture for p in sublist]
wait_pressures = [p for sublist in hysr_config.reference_posture for p in sublist]

for episode in range(2):
    print("EPISODE", episode)
    running = True
    nb_steps = 0
    while running:
        if nb_steps < 60:
            observation, reward, reset = hysr.step(wait_pressures)
        else:
            observation, reward, reset = hysr.step(swing_pressures)
        if not hysr_config.accelerated_time:
            waited = frequency_manager.wait()
            if waited < 0:
                print("! warning ! failed to maintain algorithm frequency")
        if reset:
            print("\treward:", reward)
        running = not reset
        nb_steps += 1
    hysr.reset()
    frequency_manager = o80.FrequencyManager(1.0 / algo_time_step)

hysr.close()




# # The reset method is called at the beginning of an episode
# obs = testenv.reset()
# print("Base obs = ", obs)
# # Sample a random action

# for i in range(100):
#     print("Obs = ",obs)
#     # action = testenv.action_space.sample()
#     action = np.array([1,1,1,1,1,1,1,1])
#     print("Sampled action:", action)
#     obs, reward, done, info = testenv.step(action)
#     # Note the obs is a numpy array
#     # info is an empty dict for now but can contain any debugging info
#     # reward is a scalar
#     # print(reward, done, info)
    