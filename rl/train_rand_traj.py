#%%
import os,sys

#only if using interpreter
# os.chdir('/Users/rsonker001/Documents/Personal/franka_rl_control')
sys.path.append(".")

import numpy as np

print("Current Dir = ",os.getcwd())

import wandb
from alr_sim.gyms.gym_controllers2 import GymFeedforwardController
from alr_sim.sims.SimFactory import SimRepository
from envs.tracking_env.tracking_env import TrackingEnv
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from envs.tracking_env.tests.wrappers import NormalizeActionWrapper
from utils.helper import *
from matplotlib import pyplot as plt

from stable_baselines3 import PPO
from utils.wrappers import NormalizeActionWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from datetime import datetime

NSUBSTEPS = 1
NUM_EVALS = 5 #fixed
TRAIN_MODEL = True
LOAD_MODEL = False
#DT IS MANUALLY DEFINED IN CODE , CHECK
# N_ENVS = 8#8c
EPOCHS= 70 #100
num_ep_training = 40 #40
SEED = np.random.randint(10)
log = True
DEVICE = 'cpu'
gravity = True
# reward_params = [1,1]
# reward_type = 'acc' #pos or #acc
NOISE = 1.0
TARGET_KL = 10.0
LOG_STD = -1
PLOTS = False
# NETWORK = '' 
mbfb = False

# log_dir = "local_logs" #local
log_dir = "/home/temp_store/rohit_sonker/" #remote
# log_dir = "/home/rohit/Documents/local_logs/"

####################################################
# AUTOMATED RUNS  !

import itertools
outer_loop_test = {
    "reward_type":['fb_linearized'],
    "reward_params" : [[1,0.0]],
    "noise":[0.0],
    "dt":[0.001],
    "gain_factor" : [1]
}
keys, values = zip(*outer_loop_test.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

####################################################

for i,combination in enumerate(permutations_dicts):
    
    print("\n\nCombination Number = ", i ,"Doing combination  = ", combination,"\n\n")
    reward_type = combination['reward_type']
    reward_params = combination['reward_params']
    NOISE = combination['noise']
    SEED = np.random.randint(10)
    GAMMA = 0.99 if reward_type =='pos' else 0.0
    select_dt = combination['dt']
    EPISODE_LENGTH = 5000 / (select_dt/0.001)
    gain_factor = combination['gain_factor']
    
    #wandb
    log_dir_wandb = os.path.join(log_dir, "wandb")
    
    # project_name = "multi_traj_mbfb_dt_gains_updated"#org
    project_name = "multi_traj_mbfb_dt_gains_fresh"
    exp_name = "mbfb_"+str(mbfb) + "_dt_" + str(select_dt)+"_rew_"+reward_type + "_noise_"+str(NOISE)+'_2'
    
    config = dict(
                # n_envs = N_ENVS,
                gamma = GAMMA,
                n_evals = NUM_EVALS,
                episodes_per_training = num_ep_training,
                seed = SEED,
                gravity = gravity,
                reward_params = reward_params,
                reward_type = reward_type,
                noise = NOISE,
                target_kl = TARGET_KL,
                dt = select_dt,
                gain_factor = gain_factor,
                mbfb = mbfb
                # log_std_init = LOG_STD
                )

    #tensorboard
    log_dir_tensorboard = os.path.join(log_dir, "tensorboard","ppo_panda_tracking")

    save_dir = "rl_panda_experiments/models/tracking/" + project_name + '/' + exp_name + '/'
    # save_dir = "rl_panda_experiments/models/tracking/PPO/last_model/"
    stats_path = os.path.join(save_dir, "vec_normalize.pkl")

    org_traj = np.load("tests/traj_grav_comp_logs.npz")

    def eval_model(env, model, render = False, deterministic=True, gamma=0.99, num_episodes = 3, predict_zero = False):
        
        all_rewards = []
        logs_list2 = []
        env.env_method("reset_eval_generator")
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
                logs_list1.append(logs[0])
                # if done:
                #     obs = env.reset()
            
            logs_list2.append(logs_list1)    

            ep_reward = rewards
                
            sum_ep_reward = np.sum(ep_reward)
            all_rewards.append(sum_ep_reward)

        all_rewards = np.array(all_rewards)
        return np.mean(all_rewards), np.std(all_rewards), all_rewards, logs_list2


    def random_actions(env):
        obs = env.reset()
        done = False
        for i in range(EPISODE_LENGTH):
                action = env.action_space.sample()
                obs, reward, done, _ = env.step(action)
                print("Timestep {} : Reward = {}".format(i,reward))
                if done: #this is problematic for vector env, define separate env for eval
                    env.reset()
                    

    #%% Set up environment

    #Single env

    simulator = "mujoco"
    sim_factory = SimRepository.get_factory(simulator)
    scene = sim_factory.create_scene(render=sim_factory.RenderMode.BLIND, dt = select_dt)
    r = sim_factory.create_robot(scene)
    cntrl = GymFeedforwardController(robot = r, noise=NOISE,dt = select_dt, gain_factor = gain_factor, mbfb=mbfb)
    env = TrackingEnv(
        scene = scene, robot = r, controller = cntrl, random_env = True,
        reward_coeff=reward_params, reward_type=reward_type, feedforward=True,
    )
    env.start()
    env = Monitor(env)
    env = NormalizeActionWrapper(env)
    env = DummyVecEnv([lambda:env])

    # Multiple envs

    # env = DummyVecEnv([make_env_panda(i) for i in range(N_ENVS)]) 

    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=GAMMA)
    print("Observation Space = ",env.observation_space)
    print("Action Space = ", env.action_space)

    #eval env
    simulator = "mujoco"
    sim_factory2 = SimRepository.get_factory(simulator)
    scene2 = sim_factory2.create_scene(render=sim_factory2.RenderMode.BLIND,dt = select_dt)
    r2 = sim_factory2.create_robot(scene2)
    r2.gravity_comp = gravity
    
    cntrl2 = GymFeedforwardController(robot = r2, noise=NOISE,dt = select_dt, gain_factor = gain_factor, mbfb=mbfb)
    eval_env = TrackingEnv(
        scene = scene2, robot = r2, controller = cntrl2, random_env = True, robot_logs=True,
        reward_coeff=reward_params, reward_type=reward_type, feedforward=True,
        eval_mode=True
    )

    eval_env.start()

    eval_env = NormalizeActionWrapper(eval_env)
    eval_env = DummyVecEnv([lambda:eval_env])
    eval_env.seed(SEED)
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


    if TRAIN_MODEL and LOAD_MODEL != True:
        # will reset normalization
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, gamma=GAMMA)
        eval_env.training=False
        eval_env.norm_reward = False
        
    if LOAD_MODEL:
        #loads normalization constants
        eval_env = VecNormalize.load(stats_path, eval_env) #load stats
        eval_env.training=False
        eval_env.norm_reward = False

        # Load the agent
        model = PPO.load(save_dir + "ppo_panda_tracking_random", env=env, device=DEVICE)


    # %% 
    # Random Agent, before training
    # print("Random Agent")

    # #vizualise using own function
    # ep_reward = eval_model(eval_env, model, render=False, gamma=GAMMA)
    # print("Episode reward = ",ep_reward)
    
    print("\n\n With zero baseline")
    ep_reward, ep_std, all_rewards, logs_list = eval_model(eval_env, model, render=False, gamma=GAMMA, num_episodes = NUM_EVALS, predict_zero=True)
    print("Episode reward = ",ep_reward," . std dev = ",ep_std)
    print("All rewards = ",all_rewards)
    
    pos_error = []
    fb_lin_error =[]
    
    for ev in range(NUM_EVALS):
        logs = unpack_logs(logs_list, index=ev)
        pos_tracking_err, acc_tracking_err, ff_norm, command_rl,fb_lin_err = compute_tracking_error(logs)
        pos_error.append(pos_tracking_err)
        fb_lin_error.append(fb_lin_err)
        print("\nTrajectory : ",ev)
        print("Pos Tracking Error = ",pos_tracking_err)
        print("Acc Tracking Error = ",acc_tracking_err)
        print("FB linearization error = ",np.mean(fb_lin_err))
        print("FF avg norm = ",ff_norm)
        print("FF component sum of norms",np.sum(np.linalg.norm(command_rl, axis=1)))

        config['baseline_reward_'+str(ev+1)] = all_rewards[ev]
        config['baseline_pos_error_'+str(ev+1)] = pos_tracking_err
    
    config['baseline_reward_mean'] = ep_reward
    config['baseline_pos_error_mean'] = np.mean(pos_error)
    config['baseline_fb_lin_error_mean'] = np.mean(fb_lin_error)
    
    # ep_reward, ep_std, all_rewards, logs_list = eval_model(eval_env, model, render=False, gamma=GAMMA, num_episodes = NUM_EVALS, predict_zero=True)
    print("Episode reward = ",ep_reward," . std dev = ",ep_std)
    print("All rewards = ",all_rewards)
    
    # global global_torque_norm_limit 
    # global_torque_norm_limit = 1.5 * pos_tracking_err
    
    # %% Training


    max_val_reward = -np.inf
    best_pos_error = np.inf
    best_acc_error = np.inf
    total_timesteps = num_ep_training*EPISODE_LENGTH
    plot_epoch = 10

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
            # ep_reward, ep_rew_var,x,_ = eval_model(env, model, render=False, gamma=GAMMA, num_episodes = NUM_EVALS)
            ep_reward, ep_rew_var, all_rewards, logs_list = eval_model(eval_env, model, render=False, gamma=GAMMA, num_episodes = NUM_EVALS)
            t2 = datetime.now()-t1
            
            pos_error =[]
            norm_ff_arr = []
            acc_error = []
            logs_array =[]
            fb_lin_arr = []
            
            for ev in range(NUM_EVALS):
                logs = unpack_logs(logs_list, index=ev) 
                pos_tracking_err, acc_tracking_err, norm_ff, command_rl, fb_lin_err = compute_tracking_error(logs)
                pos_error.append(pos_tracking_err)
                acc_error.append(acc_tracking_err)
                norm_ff_arr.append(norm_ff)
                logs_array.append(logs)
                fb_lin_arr.append(fb_lin_err)
                # ep1_reward = all_rewards[0]
                # rew_torque_penalty =  - reward_params[1] * np.sum(np.linalg.norm(command_rl, axis=1))
                # rew_term1 = reward_params[0] * ep1_reward - rew_torque_penalty
            
            pos_tracking_err = np.mean(pos_error)
            fb_lin_err = np.mean(fb_lin_arr)
            
            print("Iteration : ",i," . Mean Episode reward = ",ep_reward," . Std dev = ", ep_rew_var, " . Took time = ",t2)
            if log:
                wandb.log({"epoch":i, 
                        "eval_reward_mean":ep_reward,
                        "eval_reward_std":ep_rew_var,
                        "pos_tracking_error": np.mean(pos_error),
                        "acc_tracking_error": np.mean(acc_error),
                        "norm_ff":np.mean(norm_ff_arr),
                        "fb_lin_error":np.mean(fb_lin_arr),
                        "traj1": pos_error[0],
                        "traj2":pos_error[1],
                        "traj3":pos_error[2],
                        "traj4":pos_error[3],
                        "traj5":pos_error[4],
                        'last_train_reward':last_training_rew,
                        # "reward_term1":rew_term1,
                        # "reward_torque_penalty":rew_torque_penalty
                        })
                
            if ep_reward>max_val_reward:
                print("Saving model")
                max_val_reward = ep_reward
                model.save(save_dir + "ppo_panda_tracking_random")
                stats_path = os.path.join(save_dir, "vec_normalize.pkl")
                env.save(stats_path)
                if log:
                    wandb.summary["best_eval_reward"] = max_val_reward
                
            if best_pos_error>pos_tracking_err:
                best_pos_error = pos_tracking_err    
                if log:
                    wandb.summary['best_pos_error'] = best_pos_error
                    wandb.summary['reward_on_best_pos_error'] = ep_reward
                    wandb.summary['b_pos_error_t1'] = pos_error[0]
                    wandb.summary['b_pos_error_t2'] = pos_error[1]
                    wandb.summary['b_pos_error_t3'] = pos_error[2]
                    wandb.summary['b_pos_error_t4'] = pos_error[3]
                    wandb.summary['b_pos_error_t5'] = pos_error[4]
                    
                    if PLOTS:
                        for ev in range(NUM_EVALS):
                            # fig = plot_torque(joint_pos = logs_array[ev]['joint_pos'], des_pos = logs_array[ev]['des_joint_pos'])
                            fig = plot_error(series1 = logs_array[ev]['joint_pos'], series2 = logs_array[ev]['des_joint_pos'], label = "joint_tracking_error")
                            wandb.log({'best_pos_tracking_err_plot_'+'traj'+str(ev):fig})
                            del fig
                            
                            fig = plot_error(series1 = logs_array[ev]['joint_pos'], series2 = logs_array[ev]['des_joint_pos'], label = "joint tracking err")
                            wandb.log({'pos_tracking_error':fig})
                            del fig
                            
                            joint_acc = np.diff(logs_array[0]['joint_vel'], axis = 0)/0.001 #### DT !
                            pd_command = logs_array[ev]['pd_command'][:-1,:]
                            fig = plot_error(series1 = joint_acc, series2 = pd_command, label = "fb_lin_error")
                            wandb.log({'best_fb_lin_err_plot_'+'traj'+str(ev):fig})
                            del fig
                    
            if best_acc_error>acc_tracking_err:
                best_acc_error = acc_tracking_err
                if log:
                    wandb.summary['best_acc_error'] = best_acc_error
                    
            if PLOTS and i%plot_epoch==0:
                for ev in range(NUM_EVALS):
                    # fig = plot_torque(joint_pos = logs_array[ev]['joint_pos'], des_pos = logs_array[ev]['des_joint_pos'])
                    fig = plot_error(series1 = logs_array[ev]['joint_pos'], series2 = logs_array[ev]['des_joint_pos'], label = "joint_tracking_error")
                    wandb.log({'pos_tracking_err_plot_'+'traj'+str(ev):fig})
                    del fig
                    
                    fig = plot_error(series1 = logs_array[ev]['joint_pos'], series2 = logs_array[ev]['des_joint_pos'], label = "joint tracking err")
                    wandb.log({'pos_tracking_error':fig})
                    del fig
                    
                    joint_acc = np.diff(logs_array[0]['joint_vel'], axis = 0)/0.001 #### DT !
                    pd_command = logs_array[ev]['pd_command'][:-1,:]
                    fig = plot_error(series1 = joint_acc, series2 = pd_command, label = "fb_lin_error")
                    wandb.log({'fb_lin_err_plot_'+'traj'+str(ev):fig})
                    del fig
                    
            #save last model
            model.save(save_dir + "last_model/ppo_panda_tracking_random")
            stats_path = os.path.join(save_dir, "last_model/vec_normalize.pkl")
            env.save(stats_path)
        
        if log:
            run.finish()

    total_time = datetime.now() - t0

    print("\n\n Training complete, total time taken = ",total_time)
    
