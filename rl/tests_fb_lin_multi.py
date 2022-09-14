#%%
import time
import os, sys

#only if using interpreter
# os.chdir('/Users/rsonker001/Documents/Personal/franka_rl_control')
# os.chdir('/home/i53/student/rohit_sonker/franka_rl_control')
sys.path.append(".")

import numpy as np

from alr_sim.gyms.gym_controllers2 import GymFeedforwardController
from alr_sim.sims.SimFactory import SimRepository
from envs.tracking_env.tracking_env import TrackingEnv
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from envs.tracking_env.tests.wrappers import NormalizeActionWrapper
from utils.helper import *
from matplotlib import pyplot as plt
import time, pickle

select_dt = 0.001
EPISODE_LENGTH = 5000 / (select_dt/0.001)
NUM_EVALS=5
mbfb = False
save_result_logs_directory = "rl_results"

def eval_model(env, model, render = False, deterministic=True, gamma=0.99, num_episodes = 3, predict_zero = False):
        
    all_rewards = []
    logs_list2 = []
    env.env_method("reset_eval_generator")
    obs = env.reset()
    t = time.process_time()
    for i in range(num_episodes):
        # print("eval run ",i)
        rewards = []
        time_taken_list = []
        # obs = env.reset() #gym wrapper has already reset it
        done = False
        logs_list1 = []
        while not done:
            # if render:
            #     env.render()
            if predict_zero or np.isnan(obs).any():
                action = np.array([0,0,0,0,0,0,0]).reshape(1,7)
            else:
                if np.isnan(obs).any():
                        print("****------------- Nan obs detected, predicting zero ff action----------****")
                action, _ = model.predict(obs, deterministic=deterministic)
                if type(action)==tuple:
                    action = action[0]
            
            # print(action)
            tt = time.process_time() - t
            time_taken_list.append(tt)
            obs, r, done, logs = env.step(action)
            t = time.process_time()
            rewards.append(r)
            logs_list1.append(logs[0])
            # if done:
            #     obs = env.reset()
        
        logs_list2.append(logs_list1)    

        ep_reward = rewards
        time_taken = np.mean(np.array(time_taken_list))
        print("Time taken average = ", time_taken)
        sum_ep_reward = np.sum(ep_reward)
        all_rewards.append(sum_ep_reward)

    all_rewards = np.array(all_rewards)
    return np.mean(all_rewards), np.std(all_rewards), all_rewards, logs_list2

from stable_baselines3 import PPO

simulator = "mujoco"

sim_factory = SimRepository.get_factory(simulator)

NOISE = 0.0
reward_type = 'fb_linearized'
# project_name = "multi_traj_mbfb_dt_gains_updated" #old results
project_name = "multi_traj_mbfb_dt_gains_deploy"

exp_name = "mbfb_"+str(mbfb) + "_dt_" + str(select_dt)+"_rew_"+reward_type + "_noise_"+str(NOISE) +'_reset'
model_to_use = 'best'

if model_to_use=='best':    
    save_dir = "rl_panda_experiments/models/tracking/" + project_name + '/' + exp_name + '/'
    stats_path = os.path.join(save_dir, "vec_normalize.pkl")
else:
    save_dir = "rl_panda_experiments/models/tracking/" + project_name + '/' + exp_name + '/last_model/'
    stats_path = os.path.join(save_dir, "vec_normalize.pkl")

scene = sim_factory.create_scene(render=sim_factory.RenderMode.BLIND, dt = select_dt)
r = sim_factory.create_robot(scene)

cntrl = GymFeedforwardController(robot = r, noise=NOISE, dt = select_dt, mbfb = mbfb)

env = TrackingEnv(
        scene = scene, robot = r, controller = cntrl, max_steps_per_episode=EPISODE_LENGTH,
        random_env = True, robot_logs=True, 
        reward_type="fb_linearized", feedforward=True,reward_coeff=[1,0.0],
        eval_mode=True
    )

env.start()

env = NormalizeActionWrapper(env)
env = DummyVecEnv([lambda:env]) 

env = VecNormalize.load(stats_path, env) #load stats
env.training=False
env.norm_reward = False
env.seed(0)

model = PPO.load(save_dir + "ppo_panda_tracking_random", env=env, device='cpu')


plot_trajectory_index = 0

print("\n\n With zero baseline")
ep_reward, ep_std, all_rewards, logs_list = eval_model(env, model, render=False, gamma=0.0, num_episodes = NUM_EVALS, predict_zero=True)
print("Episode reward = ",ep_reward," . std dev = ",ep_std)
print("All rewards = ",all_rewards)

pos_error = []
fb_lin_error =[]
jointwise_error_all = []

#save log list
if save_result_logs_directory is not None:
    if not os.path.exists(save_result_logs_directory):
        os.makedirs(save_result_logs_directory)
    with open(os.path.join(save_result_logs_directory, "baseline.pkl"), 'wb') as f:
        pickle.dump(logs_list, f)

for ev in range(NUM_EVALS):
    logs = unpack_logs(logs_list, index=ev)
    pos_tracking_err, acc_tracking_err, ff_norm, command_rl,fb_lin_err = compute_tracking_error(logs)
    pos_error.append(pos_tracking_err)
    fb_lin_error.append(fb_lin_err)
    print("\nTrajectory : ",ev)
    print(f"Pos Tracking Error = {pos_tracking_err:.6f}")
    print("Acc Tracking Error = ",acc_tracking_err)
    print("FB linearization error = ",np.mean(fb_lin_err))
    print("FF avg norm = ",ff_norm)
    print("FF component sum of norms",np.sum(np.linalg.norm(command_rl, axis=1)))
    
    joint_tracking_error = np.sqrt(np.mean((logs['joint_pos'] - logs['des_joint_pos'])**2, axis=0))
    jointwise_error_all.append(joint_tracking_error)

    print("Jointwise error = ",joint_tracking_error)
    print(f"RMSE1 (mean of jointwise error) = {joint_tracking_error.mean():.6f}")

jointwise_error_all = np.array(jointwise_error_all)
logs = unpack_logs(logs_list, index=plot_trajectory_index)
joint_tracking_error = logs['joint_pos'] - logs['des_joint_pos']
#extract PD command
pd_commands = []
joint_vel = []
for i in range(len(logs_list[plot_trajectory_index])):
    pd_commands.append(logs_list[plot_trajectory_index][i]['pd_command'])
    joint_vel.append(logs_list[plot_trajectory_index][i]['joint_vel'])

pd_commands = np.array(pd_commands)
joint_vel = np.array(joint_vel)
joint_acc = np.diff(joint_vel, axis=0)/select_dt # acc[i] = vel[i+1] - vel[i] hence it is already of next timestep
error = joint_acc - pd_commands[:-1]
predictions = logs['command'].reshape(-1,7)

# config['baseline_reward_mean'] = ep_reward
# config['baseline_pos_error_mean'] = np.mean(pos_error)
# config['baseline_fb_lin_error_mean'] = np.mean(fb_lin_error)

# plot_torque(plot_name = "test_plot.png", actual = logs['joint_pos'], desired = logs['des_joint_pos'])
# plot_torque(plot_name = "test_plot_vel.png", vel = logs['joint_vel'])


print("\n\n With MODEL")
ep_reward, ep_std, all_rewards, logs_list = eval_model(env, model, render=False, gamma=0.0, num_episodes = NUM_EVALS)
print("Episode reward = ",ep_reward," . std dev = ",ep_std)
print("All rewards = ",all_rewards)

model_pos_error = []
fb_lin_error =[]
model_jointwise_error_all =[]

#save log list
if save_result_logs_directory is not None:
    if not os.path.exists(save_result_logs_directory):
        os.makedirs(save_result_logs_directory)
    with open(os.path.join(save_result_logs_directory, "with_model.pkl"), 'wb') as f:
        pickle.dump(logs_list, f)

for ev in range(NUM_EVALS):
    logs = unpack_logs(logs_list, index=ev)
    pos_tracking_err, acc_tracking_err, ff_norm, command_rl,fb_lin_err = compute_tracking_error(logs)
    model_pos_error.append(pos_tracking_err)
    fb_lin_error.append(fb_lin_err)
    print("\nTrajectory : ",ev)
    print("Pos Tracking Error = ",pos_tracking_err)
    print("Acc Tracking Error = ",acc_tracking_err)
    print("FB linearization error = ",np.mean(fb_lin_err))
    print("FF avg norm = ",ff_norm)
    print("FF component sum of norms",np.sum(np.linalg.norm(command_rl, axis=1)))
    
    joint_tracking_error = np.sqrt(np.mean((logs['joint_pos'] - logs['des_joint_pos'])**2, axis=0))
    model_jointwise_error_all.append(joint_tracking_error)

    print("Jointwise error = ",joint_tracking_error)
    print(f"RMSE1 (mean of jointwise error) = {joint_tracking_error.mean():.6f}")

model_jointwise_error_all = np.array(model_jointwise_error_all)
logs = unpack_logs(logs_list, index=plot_trajectory_index)
joint_tracking_error_model = logs['joint_pos'] - logs['des_joint_pos']
#extract PD command
pd_commands = []
joint_vel = []
for i in range(len(logs_list[plot_trajectory_index])):
    pd_commands.append(logs_list[plot_trajectory_index][i]['pd_command'])
    joint_vel.append(logs_list[plot_trajectory_index][i]['joint_vel'])

pd_commands = np.array(pd_commands)
joint_vel = np.array(joint_vel)
joint_acc = np.diff(joint_vel, axis=0)/select_dt # acc[i] = vel[i+1] - vel[i] hence it is already of next timestep
error_model = joint_acc - pd_commands[:-1]

#COMPARISON

print("************Comparison Summary**********")

# print(f"Best mean over all joint rMSE = {best_error1:.6f} in comparison to baseline of = {baseline_error1:.6f} | Improvement = {((baseline_error1 - best_error1)/baseline_error1) *100:.2f}%")
# print(f"Best Root Mean of all errors = {best_error2:.6f} in comparison to baseline of = {baseline_error2:6f} | Improvement = {((baseline_error2 - best_error2)/baseline_error2) *100:.2f}%")

# print(f"Joint wise improvement in absolute= {(baseline_eval_joint_error - best_error_joints)}")
# print(f"Joint wise improvement in percent= {(baseline_eval_joint_error - best_error_joints)/baseline_eval_joint_error *100} % ")

print("\n---- EVAL trajectory wise -----")
for i in range(NUM_EVALS):
    baseline = jointwise_error_all[i,:].mean()
    best = model_jointwise_error_all[i,:].mean()
    print(f"\nTraj {i}, RMSE1 is {best:.6f} compared to baseline {baseline:.6f} | Improvement = {(baseline-best)/baseline *100} %")
    
    baseline = pos_error[i]
    best = model_pos_error[i]
    print(f"Traj {i}, RMSE2 is {best:.6f} compared to baseline {baseline:.6f} | Improvement = {(baseline-best)/baseline *100} %")
    
    print(f"Traj {i}, Joint wise error difference = {jointwise_error_all[i,:] - model_jointwise_error_all[i,:]}")
