#%%
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

from stable_baselines3 import PPO
from basic_env.wrappers import NormalizeActionWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy

from learning_table_tennis_from_scratch.hysr_one_ball import (
    HysrOneBall,
    HysrOneBallConfig,
)

import logging
from basic_env.one_ball_alt import HysrOneBall_single_robot
from learning_table_tennis_from_scratch.rewards import JsonReward
from basic_env.plotting import plot_values

NUM_EVALS = 1
TRAIN_MODEL = True
# LOAD_MODEL = False
EPOCHS= 10 #100
num_ep_training = 1 #40
SEED = np.random.randint(10)
log = True
DEVICE = 'cpu'

TARGET_KL = 10.0
LOG_STD = -1
PLOTS = False


log_dir = "local_logs" #local
# log_dir = "/home/temp_store/rohit_sonker/" #remote
# log_dir = "/home/rohit/Documents/local_logs/"

####################################################
# AUTOMATED RUNS  !

# import itertools
# outer_loop_test = {
#     "reward_type":['fb_linearized'],
#     "reward_params" : [[1,0.0]],
#     "noise":[0.0],
#     "dt":[0.001],
#     "gain_factor" : [1]
# }
# keys, values = zip(*outer_loop_test.items())
# permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

####################################################

# for i,combination in enumerate(permutations_dicts):
    
# print("\n\nCombination Number = ", i ,"Doing combination  = ", combination,"\n\n")
# reward_type = combination['reward_type']
# reward_params = combination['reward_params']
# NOISE = combination['noise']

GAMMA = 0.0

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
                action = np.array([0,0,0,0,0,0,0]).reshape(1,7)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            
            # print(action)
            obs, r, done, logs = env.step(action)
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
    return np.mean(all_rewards), np.std(all_rewards), all_rewards, logs_list2


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

# def compute_tracking_errors(logs):
#     all_tracking_error = logs['joint_pos'][1:] - logs['joint_pos_des'][:-1]
#     joint_tracking_error = np.sqrt(np.mean((logs['joint_pos'][1:] - logs['joint_pos_des'][:-1])**2, axis=0))
#     tracking_mean_error = joint_tracking_error.mean()
#     print("\n\n")
#     print(f"Pos tracking error joint-wise : ", joint_tracking_error)
#     print(f"RMSE1 (mean of jointwise error) = {joint_tracking_error.mean():.6f}")

#     acc = np.diff(logs['joint_vel'])/DT #i = i+1 - i             
#     all_fb_lin_error = acc - logs['command'][1:]
#     fb_lin_error_mean = np.sqrt(np.mean(all_fb_lin_error**2), axis=0)
#     fb_lin_mean_error = fb_lin_error_mean.mean()
#     print(f"FB Linearization error joint-wise : ", fb_lin_error_mean)
#     print(f"RMSE1 (mean of jointwise error) = {fb_lin_error_mean.mean():.6f}")

#     return tracking_mean_error, all_tracking_error, fb_lin_mean_error, all_fb_lin_error

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

#wandb
log_dir_wandb = os.path.join(log_dir, "wandb")

# project_name = "multi_traj_mbfb_dt_gains_updated"#org
project_name = "pam_single_traj"
exp_name = "ppo_dt_"+str(DT)

config = dict(
            # n_envs = N_ENVS,
            dt  = DT,
            ep_len = EPISODE_LENGTH,
            gamma = GAMMA,
            n_evals = NUM_EVALS,
            episodes_per_training = num_ep_training,
            seed = SEED,
            )

#tensorboard
log_dir_tensorboard = os.path.join(log_dir, "tensorboard","pam_single_traj")

save_dir = "rl/models/" + project_name + '/' + exp_name + '/'
# save_dir = "rl_panda_experiments/models/tracking/PPO/last_model/"
stats_path = os.path.join(save_dir, "vec_normalize.pkl")

print(
"\nusing configuration files:\n- {}\n- {}\n".format(reward_config_path, hysr_config_path)
)

env = HysrOneBall_single_robot(hysr_config, reward_function, logs = True)

env = Monitor(env)
env = NormalizeActionWrapper(env)
env = DummyVecEnv([lambda:env])

env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=GAMMA)
print("Observation Space = ",env.observation_space)
print("Action Space = ", env.action_space)

#%%


policy_kwargs = dict(
    log_std_init= LOG_STD,
    # net_arch=[dict(vf=[128, 128], pi=[64, 64])],
    # squash_output=True
    )


model = PPO('MlpPolicy', env, verbose=0, gamma=GAMMA, 
            tensorboard_log=log_dir_tensorboard, seed = SEED, device=DEVICE,
            target_kl=TARGET_KL,
            policy_kwargs=policy_kwargs
            )


# if TRAIN_MODEL and LOAD_MODEL != True:
#     # will reset normalization
#     eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, gamma=GAMMA)
#     eval_env.training=False
#     eval_env.norm_reward = False
    
# if LOAD_MODEL:
#     #loads normalization constants
#     # env = VecNormalize.load(stats_path, env) #load stats
#     # env.training=False
#     # env.norm_reward = False

#     # Load the agent
#     model = PPO.load(save_dir + "ppo_panda_tracking_random", env=env, device=DEVICE)


# %% 
# Random Agent, before training
print("Random Agent")#eval with random policy
# avg_eval_rew, std_eval_rew = evaluate_policy(model, env, n_eval_episodes = NUM_EVALS, deterministic= True)
# print("Average reward with random policy = ",avg_eval_rew)   

## Random agent manual eval
avg_eval_rew, std_eval_rew, all_rewards, logs_list = eval_model(env, model, render=False, gamma=GAMMA, num_episodes = NUM_EVALS)

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
    fb_lin_error_mean = np.sqrt(np.mean(all_fb_lin_error**2, axis=0))
    fb_lin_mean_error = fb_lin_error_mean.mean()
    fb_lin_arr.append(fb_lin_mean_error)
    print(f"FB Linearization error joint-wise : ", fb_lin_error_mean)
    print(f"FB RMSE1 (mean of jointwise FB error) = {fb_lin_error_mean.mean():.6f}")

    actions = logs['actions']
    action_norm = np.linalg.norm(actions, axis=0)
    print(f"FF norm = ",action_norm)

plot_values(plot_name = f"rand_pos_error",dof = 4, pos_error = all_tracking_error)
plot_values(plot_name = f"rand_fb_lin_error",dof = 4, fb_lin_error = all_fb_lin_error)
plot_values(plot_name = f"rand_pid", dof = 4, pid_commands = logs['pid_command'][1:])
plot_values(plot_name = f"rand_reward",dof = 1, rewards = logs['rewards'].reshape(-1,1))
plot_values(plot_name = "rand_action", dof = 4, actions = actions)

pos_tracking_err = np.mean(pos_error)
fb_lin_err = np.mean(fb_lin_arr)

print("Rand "," . Mean Episode reward = ",avg_eval_rew," . Std dev = ", std_eval_rew)

# %% Training
max_val_reward = -np.inf
best_pos_error = np.inf
best_acc_error = np.inf
total_timesteps = num_ep_training*EPISODE_LENGTH
plot_epoch = 2

# t0 = time.time()
t0 = datetime.now()
print("\n\nStarting Training....")

if TRAIN_MODEL:
    
    if log:
        run = wandb.init(project=project_name, name=exp_name, dir = log_dir_wandb, config=config, sync_tensorboard=True)


    for i in range(EPOCHS):
        t1=datetime.now()
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        last_training_rew = env.get_attr('episode_returns')[0][-1]
        # avg_eval_rew, std_eval_rew = evaluate_policy(model, env, n_eval_episodes = NUM_EVALS, deterministic= True)
        
        # # EVAL and corresponding saves
        avg_eval_rew, std_eval_rew, all_rewards, logs_list = eval_model(env, model, render=False, gamma=GAMMA, num_episodes = NUM_EVALS)
        t2 = datetime.now()-t1
        
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
            fb_lin_error_mean = np.sqrt(np.mean(all_fb_lin_error**2, axis=0))
            fb_lin_mean_error = fb_lin_error_mean.mean()
            fb_lin_arr.append(fb_lin_mean_error)
            print(f"FB Linearization error joint-wise : ", fb_lin_error_mean)
            print(f"FB RMSE1 (mean of jointwise error) = {fb_lin_error_mean.mean():.6f}")

            actions = logs['actions']
            action_norm = np.linalg.norm(actions, axis=0)
            print(f"FF norm = ",action_norm)
        
        if i%plot_epoch==0:
            plot_values(plot_name = f"iter_{i}_pos_error",dof = 4, error = all_tracking_error)
            plot_values(plot_name = f"iter_{i}_fb_lin_error",dof = 4, error = all_fb_lin_error)
            plot_values(plot_name = f"iter_{i}_pid", dof = 4, pid_commands = logs['pid_command'][1:])
            plot_values(plot_name = f"iter_{i}_reward",dof = 1, rewards = logs['rewards'].reshape(-1,1))
            plot_values(plot_name = f"iter_{i}_actoin", dof = 4, action = actions)

        pos_tracking_err = np.mean(pos_error)
        fb_lin_err = np.mean(fb_lin_arr)
        
        print("Iteration : ",i," . Mean Episode reward = ",avg_eval_rew," . Std dev = ", std_eval_rew, " . Took time = ",t2)
        # print(f"Iteration {i} : Last training reward = {last_training_rew} : Eval rew = {avg_eval_rew}")
        if log:
            wandb.log({"epoch":i, 
                    # "eval_reward_mean":ep_reward,
                    # "eval_reward_std":ep_rew_var,
                    "pos_tracking_error": pos_tracking_err,
                    # "acc_tracking_error": np.mean(acc_error),
                    "norm_action":action_norm,
                    "fb_lin_error":fb_lin_err,
                    # "traj1": pos_error[0],
                    # "traj2":pos_error[1],
                    # "traj3":pos_error[2],
                    # "traj4":pos_error[3],
                    # "traj5":pos_error[4],
                    'last_train_reward':last_training_rew,
                    'avg_eval_reward':avg_eval_rew,
                    'std_eval_reward':std_eval_rew,
                    # 'avg_eval_reward':avg_eval_rew,
                    # 'std_eval_reward':std_eval_rew,
                    # "reward_term1":rew_term1,
                    # "reward_torque_penalty":rew_torque_penalty
                    })
            
        if avg_eval_rew>max_val_reward:
            print("Saving model")
            max_val_reward = avg_eval_rew
            model.save(save_dir + "ppo_pam_single_traj")
            stats_path = os.path.join(save_dir, "vec_normalize.pkl")
            env.save(stats_path)
            if log:
                wandb.summary["best_train_reward"] = max_val_reward
            
        # if best_pos_error>pos_tracking_err:
        #     best_pos_error = pos_tracking_err    
        #     if log:
        #         wandb.summary['best_pos_error'] = best_pos_error
        #         wandb.summary['reward_on_best_pos_error'] = ep_reward
        #         wandb.summary['b_pos_error_t1'] = pos_error[0]
        #         wandb.summary['b_pos_error_t2'] = pos_error[1]
        #         wandb.summary['b_pos_error_t3'] = pos_error[2]
        #         wandb.summary['b_pos_error_t4'] = pos_error[3]
        #         wandb.summary['b_pos_error_t5'] = pos_error[4]
                
        #         if PLOTS:
        #             for ev in range(NUM_EVALS):
        #                 # fig = plot_torque(joint_pos = logs_array[ev]['joint_pos'], des_pos = logs_array[ev]['des_joint_pos'])
        #                 fig = plot_error(series1 = logs_array[ev]['joint_pos'], series2 = logs_array[ev]['des_joint_pos'], label = "joint_tracking_error")
        #                 wandb.log({'best_pos_tracking_err_plot_'+'traj'+str(ev):fig})
        #                 del fig
                        
        #                 fig = plot_error(series1 = logs_array[ev]['joint_pos'], series2 = logs_array[ev]['des_joint_pos'], label = "joint tracking err")
        #                 wandb.log({'pos_tracking_error':fig})
        #                 del fig
                        
        #                 joint_acc = np.diff(logs_array[0]['joint_vel'], axis = 0)/0.001 #### DT !
        #                 pd_command = logs_array[ev]['pd_command'][:-1,:]
        #                 fig = plot_error(series1 = joint_acc, series2 = pd_command, label = "fb_lin_error")
        #                 wandb.log({'best_fb_lin_err_plot_'+'traj'+str(ev):fig})
        #                 del fig
                
        # if best_acc_error>acc_tracking_err:
        #     best_acc_error = acc_tracking_err
        #     if log:
        #         wandb.summary['best_acc_error'] = best_acc_error
                
        # if PLOTS and i%plot_epoch==0:
        #     for ev in range(NUM_EVALS):
        #         # fig = plot_torque(joint_pos = logs_array[ev]['joint_pos'], des_pos = logs_array[ev]['des_joint_pos'])
        #         fig = plot_error(series1 = logs_array[ev]['joint_pos'], series2 = logs_array[ev]['des_joint_pos'], label = "joint_tracking_error")
        #         wandb.log({'pos_tracking_err_plot_'+'traj'+str(ev):fig})
        #         del fig
                
        #         fig = plot_error(series1 = logs_array[ev]['joint_pos'], series2 = logs_array[ev]['des_joint_pos'], label = "joint tracking err")
        #         wandb.log({'pos_tracking_error':fig})
        #         del fig
                
        #         joint_acc = np.diff(logs_array[0]['joint_vel'], axis = 0)/0.001 #### DT !
        #         pd_command = logs_array[ev]['pd_command'][:-1,:]
        #         fig = plot_error(series1 = joint_acc, series2 = pd_command, label = "fb_lin_error")
        #         wandb.log({'fb_lin_err_plot_'+'traj'+str(ev):fig})
        #         del fig
                
        #save last model
        model.save(save_dir + "last_model/ppo_pam_single_traj")
        stats_path = os.path.join(save_dir, "last_model/vec_normalize.pkl")
        env.save(stats_path)
    
    if log:
        run.finish()

total_time = datetime.now() - t0

print("\n\n Training complete, total time taken = ",total_time)
    
