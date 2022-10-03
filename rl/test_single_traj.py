#%%
from asyncio import SafeChildWatcher
import os,sys

#only if using interpreter
# os.chdir('/Users/rsonker001/Documents/Personal/franka_rl_control')
sys.path.append(".")
sys.path.append("basic_env")

import numpy as np

print("Current Dir = ",os.getcwd())

import wandb

from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
# from utils.helper import *
from matplotlib import pyplot as plt

from stable_baselines3 import PPO, SAC
from basic_env.wrappers import NormalizeActionWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy

from learning_table_tennis_from_scratch.hysr_one_ball import (
    HysrOneBall,
    HysrOneBallConfig,
)

import warnings
warnings.filterwarnings('ignore')

import logging
from basic_env.one_ball_alt import HysrOneBall_single_robot
from learning_table_tennis_from_scratch.rewards import JsonReward
from basic_env.plotting import plot_values, return_plot
from wandb.integration.sb3 import WandbCallback
import plotly

NUM_EVALS = 1

SEED = np.random.randint(10)

DEVICE = 'cpu'
PLOTS = True

GAMMA = 0.99
ALGO = 'PPO' #CAPS
ACTION_DOMAIN = 'pressure'
REWARD_TYPE = 'pos'

# log_dir = "local_logs" #local
log_dir = "/home/temp_store/rohit_sonker/" #remote
# log_dir = "/home/rohit/Documents/local_logs/"


def eval_model(env, model, render = False, deterministic=True, gamma=0.0, num_episodes = 3, predict_zero = False):
    
    all_rewards = []
    logs_list2 = []
    # env.env_method("reset_eval_generator")
    obs = env.reset()           
    for i in range(num_episodes):
        # print("eval run ",i)
        rewards = []
        # obs = env.reset() #gym wrapper has already reset it
        done = False
        logs_list1 = []
        while not done:
            # if render:
            #     env.render()
            if predict_zero or np.isnan(obs).any():
                if np.isnan(obs).any():
                    print("****------------- Nan obs detected, predicting zero ff action----------****")
                action = np.array([0]*env.action_space.shape[0]).reshape(1,env.action_space.shape[0])
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            
            # if ACTION_DOMAIN == 'pressure':
            #     action = action.astype(int)
            # print(action)
            obs, r, done, logs = env.step(action)
            r = env.get_original_reward()
            rewards.append(r)
            logs_list1.append(logs)
            # if done:
            #     obs = env.reset()
        # logs = env.env_method("dump_logger")
        logs_list2.append(logs_list1)

        ep_reward = rewards
            
        sum_ep_reward = np.sum(ep_reward)
        all_rewards.append(sum_ep_reward)

    all_rewards = np.array(all_rewards)
    return np.mean(all_rewards), np.std(all_rewards), all_rewards, logs_list2, rewards


def unpack_logs(log_list):

    joint_pos = []
    joint_vel= []
    actions = []
    rewards = []
    pid_command = []
    command = []
    joint_pos_des = []
    joint_vel_des = []

    for x in log_list:
        items= x[0]
        joint_pos.append(items['obs_next'][:4].copy())
        joint_vel.append(items['obs_next'][4:8].copy())
        actions.append(items['action'].copy())
        rewards.append(items['reward'])
        pid_command.append(items['pid_command'].copy())
        command.append(items['full_command'].copy())
        joint_pos_des.append(items['obs_des'].copy())
        joint_vel_des.append(items['obs_vel_des'].copy())

    all_data_np = {
    'joint_pos' : np.array(joint_pos),
    'joint_vel' : np.array(joint_vel),
    'actions' : np.array(actions),
    'rewards' : np.array(rewards),
    'command' : np.array(command),
    'pid_command' : np.array(pid_command),
    'joint_pos_des' : np.array(joint_pos_des),
    'joint_vel_des':np.array(joint_vel_des)
    }

    return all_data_np


#%% Set up environment

#Single env

logging.basicConfig(format="hysr_one_ball_swing | %(message)s", level=logging.INFO)
# reward_config_path, hysr_config_path = _configure()
hysr_config_path = "config/hysr_single_robot_traj.json"
reward_config_path = "config/reward_default.json"

hysr_config = HysrOneBallConfig.from_json(hysr_config_path)
reward_function = JsonReward.get(reward_config_path)
algo_time_step = hysr_config.algo_time_step

DT = hysr_config.algo_time_step
EPISODE_LENGTH = hysr_config.nb_steps_per_episode

print(
"\nusing configuration files:\n- {}\n- {}\n".format(reward_config_path, hysr_config_path)
)

env = HysrOneBall_single_robot(hysr_config, reward_function, logs = True, 
                               reward_type=REWARD_TYPE,
                               action_domain = ACTION_DOMAIN
                               )

#wandb
log_dir_wandb = os.path.join(log_dir, "wandb")

# project_name = "multi_traj_mbfb_dt_gains_updated"#org
project_name = "pam_single_traj_dt_"+str(DT)
act_high = np.max(env.action_space.high)
exp_name = ALGO + "_acctime_"+"dom_"+ACTION_DOMAIN + "_rew_" + REWARD_TYPE + '_act_'+str(act_high)+"G_"+str(GAMMA)


save_dir = "rl/models/" + project_name + '/' + exp_name + '/'
# save_dir = "rl_panda_experiments/models/tracking/PPO/last_model/"
stats_path = os.path.join(save_dir, "vec_normalize.pkl")

print("Action Space org = ", env.action_space)

env = Monitor(env)
env = NormalizeActionWrapper(env)
env = DummyVecEnv([lambda:env])

env = VecNormalize.load(stats_path, env) #load stats
env.training=False
env.norm_reward = False


if ALGO == 'PPO':
    env = VecFrameStack(env, n_stack = 4)

print("Observation Space = ",env.observation_space)
print("Action Space = ", env.action_space)


# Load the agent
model = PPO.load(save_dir + "ppo_pam_single_traj", env=env, device=DEVICE)


# %% 
# Random Agent, before training
print("Zero Basline")#eval with random policy

# print("Average reward with random policy = ",avg_eval_rew)   

## Random agent manual eval
avg_eval_rew, std_eval_rew, all_rewards, logs_list, r = eval_model(env, model, render=False, gamma=GAMMA, num_episodes = NUM_EVALS, predict_zero=True)
# avg_eval_rew2, std_eval_rew2 = evaluate_policy(model, env, n_eval_episodes = NUM_EVALS, deterministic= True)

pos_error =[]
norm_ff_arr = []
acc_error = []
logs_array =[]
fb_lin_arr = []

for ev in range(NUM_EVALS):
    logs = unpack_logs(logs_list[ev])
    all_tracking_error = logs['joint_pos'][1:] - logs['joint_pos_des'][:-1]
    joint_tracking_error = np.sqrt(np.mean((logs['joint_pos'][1:] - logs['joint_pos_des'][:-1])**2, axis=0))
    tracking_mean_error = joint_tracking_error.mean()
    pos_error.append(tracking_mean_error)
    print("\n\n")
    print(f"Pos tracking error joint-wise : ", joint_tracking_error)
    print(f"RMSE1 (mean of jointwise error) = {joint_tracking_error.mean():.6f}")

    acc = np.diff(logs['joint_vel'],axis=0)/DT #i = i+1 - i             
    all_fb_lin_error = acc - logs['command'][1:]
    # this is axis =0 hence joint-wise computation, to match reward set axis=1 (step wise) and later sum for whole episode
    fb_lin_error_mean = np.sqrt(np.mean(all_fb_lin_error**2, axis=0))
    fb_lin_mean_error = np.mean(np.linalg.norm(all_fb_lin_error, axis=0))
    fb_lin_arr.append(fb_lin_mean_error)
    print(f"FB Linearization error joint-wise : ", fb_lin_error_mean)
    print(f"FB RMSE1 (mean of jointwise FB error) = {fb_lin_error_mean.mean():.6f}")

    zero_RMSE1 = joint_tracking_error.mean()
    zero_reward = avg_eval_rew
    zero_fb_lin_err = fb_lin_mean_error
    actions = logs['actions']
    action_norm = np.linalg.norm(actions, axis=0)
    print(f"FF norm = ",action_norm)

zero_pos_error = all_tracking_error
zero_fb_lin_error = all_fb_lin_error
zero_pid_commands = logs['pid_command'][1:]
zero_rewards = logs['rewards'].reshape(-1,1)
zero_actions = actions
# if PLOTS:
#     plot_values(plot_name = f"zero_pos_error",dof = 4, zero_pos_error = all_tracking_error)
#     plot_values(plot_name = f"zero_fb_lin_error",dof = 4, zero_fb_lin_error = all_fb_lin_error)
#     plot_values(plot_name = f"zero_pid", dof = 4, zero_pid_commands = logs['pid_command'][1:])
#     plot_values(plot_name = f"zero_reward",dof = 1, zero_rewards = logs['rewards'].reshape(-1,1))
#     plot_values(plot_name = "zero_action", dof = 4, zero_actions = actions)

pos_tracking_err = np.mean(pos_error)
fb_lin_err = np.mean(fb_lin_arr)

print("Zero baseline "," . Mean Episode reward = ",avg_eval_rew," . Std dev = ", std_eval_rew," . All_rewards = ",all_rewards)
# print("Rand eval func"," . Mean Episode reward = ",avg_eval_rew2," . Std dev = ", std_eval_rew2)


# Random Agent, before training-------------------------------------------------------------------------------------------
print("\n\nLoaded agent")#eval with random policy

# print("Average reward with random policy = ",avg_eval_rew)   

## Random agent manual eval
avg_eval_rew, std_eval_rew, all_rewards, logs_list, r = eval_model(env, model, render=False, gamma=GAMMA, num_episodes = NUM_EVALS)
avg_eval_rew2, std_eval_rew2 = evaluate_policy(model, env, n_eval_episodes = NUM_EVALS, deterministic= True)

pos_error =[]
norm_ff_arr = []
acc_error = []
logs_array =[]
fb_lin_arr = []

for ev in range(NUM_EVALS):
    logs = unpack_logs(logs_list[ev])
    all_tracking_error = logs['joint_pos'][1:] - logs['joint_pos_des'][:-1]
    joint_tracking_error = np.sqrt(np.mean((logs['joint_pos'][1:] - logs['joint_pos_des'][:-1])**2, axis=0))
    tracking_mean_error = joint_tracking_error.mean()
    pos_error.append(tracking_mean_error)
    print("\n\n")
    print(f"Pos tracking error joint-wise : ", joint_tracking_error)
    print(f"RMSE1 (mean of jointwise error) = {joint_tracking_error.mean():.6f}")

    acc = np.diff(logs['joint_vel'],axis=0)/DT #i = i+1 - i             
    all_fb_lin_error = acc - logs['command'][1:]
    # this is axis =0 hence joint-wise computation, to match reward set axis=1 (step wise) and later sum for whole episode
    fb_lin_error_mean = np.sqrt(np.mean(all_fb_lin_error**2, axis=0))
    fb_lin_mean_error = np.mean(np.linalg.norm(all_fb_lin_error, axis=0))
    fb_lin_arr.append(fb_lin_mean_error)
    print(f"FB Linearization error joint-wise : ", fb_lin_error_mean)
    print(f"FB RMSE1 (mean of jointwise FB error) = {fb_lin_error_mean.mean():.6f}")

    random_RMSE1 = joint_tracking_error.mean()
    actions = logs['actions']
    action_norm = np.linalg.norm(actions, axis=0)
    print(f"FF norm = ",action_norm)


if PLOTS:
    plot_values(plot_name = f"eval_pos_error",dof = 4, pos_error = all_tracking_error,zero_pos_error =zero_pos_error )
    plot_values(plot_name = f"eval_fb_lin_error",dof = 4, fb_lin_error = all_fb_lin_error,zero_fb_lin_error=zero_fb_lin_error)
    plot_values(plot_name = f"eval_pid", dof = 4, pid_commands = logs['pid_command'][1:], zero_pid_commands=zero_pid_commands)
    plot_values(plot_name = f"eval_reward",dof = 1, rewards = logs['rewards'].reshape(-1,1), zero_rewards=zero_rewards)
    plot_values(plot_name = "eval_action", dof = 4, actions = actions, zero_actions=zero_actions)

pos_tracking_err = np.mean(pos_error)
fb_lin_err = np.mean(fb_lin_arr)

# print("Test "," . Mean Episode reward = ",avg_eval_rew," . Std dev = ", std_eval_rew)
# print("Test eval func"," . Mean Episode reward = ",avg_eval_rew2," . Std dev = ", std_eval_rew2," . All_rewards = ",all_rewards)

print("Final "," . Mean Episode reward = ",avg_eval_rew," . Std dev = ", std_eval_rew)
print("Final eval func"," . Mean Episode reward = ",avg_eval_rew2," . Std dev = ", std_eval_rew2," . All_rewards = ",all_rewards)
print("\n")
print(f"Eval reward = {avg_eval_rew:.2f} and zero_reward = {zero_reward:.2f} | percent = {-1*(avg_eval_rew - zero_reward)*100/zero_reward:.2f} %")
print(f"Eval pos tracking err = {pos_tracking_err:.4f} and zero RMSE = {zero_RMSE1:.4f} | percent = {-1*(pos_tracking_err - zero_RMSE1)*100/zero_RMSE1:.2f} %")
print(f"Eval Fb lin err = {fb_lin_err:.4f} and zero Fb lin err = {zero_fb_lin_err:.4f} | percent = {-1*(fb_lin_err - zero_fb_lin_err)*100/zero_fb_lin_err:.2f} %")

