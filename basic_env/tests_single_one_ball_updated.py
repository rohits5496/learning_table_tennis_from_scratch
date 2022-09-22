#!/usr/bin/env python3

import logging
import o80
import context
from plotting import plot_values
from learning_table_tennis_from_scratch.hysr_one_ball import (
    HysrOneBall,
    HysrOneBallConfig,
)

from one_ball_alt import HysrOneBall_single_robot
from learning_table_tennis_from_scratch.rewards import JsonReward
from learning_table_tennis_from_scratch.jsonconfig import get_json_config
    
import numpy as np

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

# trajectory_index = 49
# hysr.set_ball_behavior(index=trajectory_index)

hysr.reset()

# # swing_posture = [[18000, 15000], [15800, 19100], [18700, 15300], [18000, 18000]]
# swing_posture = [[0,0], [0,0], [0,0], [0,0]]
# swing_pressures = [p for sublist in swing_posture for p in sublist]
# # wait_pressures = [p for sublist in hysr_config.reference_posture for p in sublist]
# wait_pressures = [0,0,0,0,0,0,0,0]

action = [0,0,0,0]
data_iter = {}
for episode in range(1):
    print("EPISODE", episode)
    running = True                                                                                                                                                                                                                                                                                                                                                                                                                               
    nb_steps = 0
    while running:
        if nb_steps < 60:
            observation, reward, reset, log = hysr.step(action)
        else:
            observation, reward, reset, log = hysr.step(action)
        print(f"{nb_steps} : Obs = {observation[:4]} | reward = {reward}")

        running = not reset
        nb_steps += 1
    logs = hysr.dump_logger()
    data_iter[episode] = logs
    hysr.reset()

    #Trajectory - Metrics
    joint_tracking_error = np.sqrt(np.mean((logs['joint_pos'][1:] - logs['joint_pos_des'][:-1])**2, axis=0))
    print("\n\n")
    print(f"Pos tracking error joint-wise : ", joint_tracking_error)
    print(f"RMSE1 (mean of jointwise error) = {joint_tracking_error.mean():.6f}")


hysr.close()

# Plots
plot_values(plot_name = 'test',dof = 4, joint_pos = logs['joint_pos'][1:], des_joint_pos = logs['joint_pos_des'][:-1])

acc = np.diff(logs['joint_vel'], axis=0)/hysr._hysr_config.algo_time_step
print("Algo timestep = ",hysr._hysr_config.algo_time_step)
plot_values(plot_name = 'test_fb_err',dof = 4, 
            pid_commands = logs['command'][1:],
            acc = acc
            )
# plot_values(dof = 1, rewards = logs['rewards'].reshape(-1,1))



