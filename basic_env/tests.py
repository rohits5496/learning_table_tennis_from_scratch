import gym
import numpy as np
import os                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
print(os.getcwd())

import sys
sys.path.append('/home/rohit/Documents/learning_table_tennis_from_scratch/')

import time
from env import PAMenv
from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv



testenv = PAMenv(hysr_one_ball_config_file = "config/hysr_demos.json")
# testenv = HysrOneBallEnv(hysr_one_ball_config_file = 'config/hysr_demos.json')


print("Observation space:", testenv.observation_space)
print("Shape:", testenv.observation_space.shape)
# Discrete(2) means that there is two discrete actions
print("Action space:", testenv.action_space)

# The reset method is called at the beginning of an episode
obs = testenv.reset()
print("Base obs = ", obs)
# Sample a random action

for i in range(500):
    print("Obs = ",obs)
    action = testenv.action_space.sample()
    # action = np.array([1,1,1,1,1,1,1,1])
    print("Sampled action:", action)
    obs, reward, done, info = testenv.step(action)
    time.sleep(0.1)
    # Note the obs is a numpy array
    # info is an empty dict for now but can contain any debugging info
    # reward is a scalar
    # print(reward, done, info)
    