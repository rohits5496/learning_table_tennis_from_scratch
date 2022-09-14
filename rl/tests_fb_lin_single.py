#%%
import time
import os, sys


#only if using interpreter
# os.chdir('/Users/rsonker001/Documents/Personal/franka_rl_control')
# os.chdir('/home/i53/student/rohit_sonker/franka_rl_control')
# os.chdir('/home/rohit/Documents/franka_rl_control')
sys.path.append(".")

import numpy as np

from alr_sim.gyms.gym_controllers2 import GymFeedforwardController
from alr_sim.sims.SimFactory import SimRepository
from envs.tracking_env.tracking_env import TrackingEnv
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from envs.tracking_env.tests.wrappers import NormalizeActionWrapper
from utils.helper import *
from matplotlib import pyplot as plt


select_dt = 0.001
EPISODE_LENGTH = 5000 / (select_dt/0.001)
NUM_EVALS=1
mbfb = True

def eval_model(env, model=None, deterministic=True, gamma=0.99, num_episodes = 1, vectorized = False):
        
    all_rewards = []
    logs_list2 = []

    for i in range(num_episodes):
        rewards = []
        obs = env.reset() #turned off lets see how this affects
        done = False
        logs_list1 = []
        while not done:
            # if render:
            #     env.render()
            if model is not None and not np.isnan(obs).any():
                action = model.predict(obs, deterministic=deterministic)
                if type(action)==tuple:
                    action = action[0]
            else:
                if np.isnan(obs).any():
                        print("****------------- Nan obs detected, predicting zero ff action----------****")
                action = np.array([0,0,0,0,0,0,0]).reshape(1,7)
            
            # print(action)
            obs, r, done, logs = env.step(action)
            # print(logs['joint_pos'])
            rewards.append(r)
            if vectorized:
                logs_list1.append(logs[0].copy())
            else:
                logs_list1.append(logs.copy())
            
                
            # if done: #turned on since outer loop reset is now off
            #     obs = env.reset()
            
        logs_list2.append(logs_list1)    
        # if render:
        #     env.close()
        ep_reward = []
        # rew_timestep = 0
        # for i in reversed(range(len(rewards))):
        #     rew_timestep = rew_timestep*gamma + rewards[i] #start from end and roll back
        #     ep_reward.append(rew_timestep)
        ep_reward = rewards
        
        sum_ep_reward = np.sum(ep_reward)
        all_rewards.append(sum_ep_reward)
    
    all_rewards = np.array(all_rewards)
    return np.mean(all_rewards), np.std(all_rewards), all_rewards, logs_list2


from stable_baselines3 import PPO

simulator = "mujoco"

sim_factory = SimRepository.get_factory(simulator)

noise = 0.0
select_dt = 0.001
reward_type = 'fb_linearized'

device = 'cpu'
project_name = "single_traj_updated"
exp_name = "mbfb_"+str(mbfb)+"_dt_"+str(select_dt)+"_noise_"+str(noise)+'_rew_'+reward_type
model_to_use = 'last'
EPISODE_LENGTH = 5000 / (select_dt/0.001)

if model_to_use=='best':    
    save_dir = "rl_panda_experiments/models/tracking/" + project_name + '/' + exp_name + '/'
    stats_path = os.path.join(save_dir, "vec_normalize.pkl")
else:
    save_dir = "rl_panda_experiments/models/tracking/" + project_name + '/' + exp_name + '/last_model/'
    stats_path = os.path.join(save_dir, "vec_normalize.pkl")

scene = sim_factory.create_scene(render=sim_factory.RenderMode.HUMAN, dt = select_dt)
r = sim_factory.create_robot(scene)

cntrl = GymFeedforwardController(robot = r, noise=noise, dt = select_dt, mbfb = mbfb)

env = TrackingEnv(
        scene = scene, robot = r, controller = cntrl, max_steps_per_episode=EPISODE_LENGTH,
        random_env = False, robot_logs=True, 
        reward_type="fb_linearized", feedforward=True,reward_coeff=[1,0.0],
        eval_mode=False
    )

env.start()


# env = NormalizeActionWrapper(env)
# # env = DummyVecEnv([lambda:env])

# env.seed(0)

# for i in range(NUM_EVALS):
    
#     reward_mean, reward_std, all_rewards, log_list= eval_model(env, gamma = 0.99, vectorized=False)
#     logs = unpack_logs(log_list)
#     pos_tracking_err, acc_tracking_err, ff_norm,command_rl,_ = compute_tracking_error(logs, dt=select_dt)
    
#     print("\nPos Tracking Error = ",pos_tracking_err)
#     print("Acc Tracking Error = ",acc_tracking_err)
#     print("FF norm = ",ff_norm)
    
#     print("Reward mean = ", reward_mean)
#     print("\n")
    
#     # r.robot_logger.stopLogging()
#     r.robot_logger.stop_logging()
#     rMSE = np.sqrt(np.mean((r.robot_logger.joint_pos - r.robot_logger.des_joint_pos)**2, axis=0))
#     print('Mean of jointwise RMSE Tracking Error: ', np.mean(rMSE))
    

env = NormalizeActionWrapper(env)
env = DummyVecEnv([lambda:env])

env = VecNormalize.load(stats_path, env) #load stats
env.training=False
env.norm_reward = False
env.seed(0)

model = PPO.load(save_dir + "ppo_panda_tracking", env=env, device='cpu')

for i in range(NUM_EVALS):
        
    reward_mean, reward_std, all_rewards, log_list= eval_model(env, model = model, vectorized = True)
    logs = unpack_logs(log_list)
    pos_tracking_err, acc_tracking_err, ff_norm,command_rl,fb_lin_err = compute_tracking_error(logs, select_dt)

    print(f"Pos Tracking Error = {pos_tracking_err:.6f}")
    print("Acc Tracking Error = ",acc_tracking_err)
    print("FB linearization error = ",np.mean(fb_lin_err))
    print("FF avg norm = ",ff_norm)
    print("FF component sum of norms",np.sum(np.linalg.norm(command_rl)))
    
    joint_tracking_error = np.sqrt(np.mean((logs['joint_pos'] - logs['des_joint_pos'])**2, axis=0))

    print("Jointwise error = ",joint_tracking_error)
    print(f"RMSE1 (mean of jointwise error) = {joint_tracking_error.mean():.6f}")
    
    print("\n")
    # plot_torque(actual = logs['joint_pos'], desired = logs['des_joint_pos'])
joint_tracking_error_model = logs['joint_pos'] - logs['des_joint_pos']
#extract PD command
pd_commands = []
joint_vel = []
# predictions = []
for i in range(len(log_list[0])):
    pd_commands.append(log_list[0][i]['pd_command'])
    joint_vel.append(log_list[0][i]['joint_vel'])
    # predictions.append(log_list[0][i]['command'])

pd_commands = np.array(pd_commands)
joint_vel = np.array(joint_vel)
joint_acc = np.diff(joint_vel, axis=0)/select_dt # acc[i] = vel[i+1] - vel[i] hence it is already of next timestep
error_model = joint_acc - pd_commands[:-1]

predictions = logs['command'].reshape(-1,7)
# predictions = np.array(predictions, dtype=object)
print("shape = ",predictions.shape)


##### repeating without model
# r.robot_logger.startLogging()
r.robot_logger.start_logging()

for i in range(NUM_EVALS):
        
    reward_mean, reward_std, all_rewards, log_list= eval_model(env, vectorized=True)
    logs = unpack_logs(log_list)
    pos_tracking_err, acc_tracking_err, ff_norm,_,_ = compute_tracking_error(logs, dt=select_dt)
    
    pos_tracking_err, acc_tracking_err, ff_norm,command_rl,fb_lin_err = compute_tracking_error(logs, select_dt)
    
    print(f"Pos Tracking Error = {pos_tracking_err:.6f}")
    print("Acc Tracking Error = ",acc_tracking_err)
    print("FB linearization error = ",np.mean(fb_lin_err))
    print("FF avg norm = ",ff_norm)
    print("FF component sum of norms",np.sum(np.linalg.norm(command_rl)))
    
    joint_tracking_error = np.sqrt(np.mean((logs['joint_pos'] - logs['des_joint_pos'])**2, axis=0))

    print("Jointwise error = ",joint_tracking_error)
    print(f"RMSE1 (mean of jointwise error) = {joint_tracking_error.mean():.6f}")
    
    print("\n")
    # plot_torque(actual = logs['joint_pos'], desired = logs['des_joint_pos'])
joint_tracking_error = logs['joint_pos'] - logs['des_joint_pos']
#extract PD command
pd_commands = []
joint_vel = []
for i in range(len(log_list[0])):
    pd_commands.append(log_list[0][i]['pd_command'])
    joint_vel.append(log_list[0][i]['joint_vel'])

pd_commands = np.array(pd_commands)
joint_vel = np.array(joint_vel)
joint_acc = np.diff(joint_vel, axis=0)/select_dt # acc[i] = vel[i+1] - vel[i] hence it is already of next timestep
error = joint_acc - pd_commands[:-1]


# %%plot

plot_torque(
    # plot_name = "noise10_fblin_error_model.png", 
    title = "fb_linearized_error",model = error_model, nominal = error)
plot_torque(
    # plot_name = "noise10_tracking_error_model.png", 
    title = "tracking_error", model = joint_tracking_error_model, nominal = joint_tracking_error)

plot_torque(
    # plot_name = "noise10_pred_PD_model.png", 
    title = "prediction and PD", predictions = predictions, pd_command = pd_commands)
