import sys
import time
import random
import o80
import context
from lightargs import BrightArgs
from hysr_one_ball import HysrOneBall


# default frequency for real/dummy robot
# o80_pam_time_step = 0.0005
# default frequency for mujoco robot
o80_pam_time_step = 0.002

mujoco_time_step = 0.002
algo_time_step = 0.01
target_position = [0.45,2.7,0.17]
reward_normalization_constant = 1.0
smash_task = True
rtt_cap = 0.2
nb_episodes = 5

reference_posture = [[20000,12000],[12000,22000],[15000,15000],[15000,15000]]
swing_posture = [[14000,22000],[14000,22000],[17000,13000],[14000,16000]]

def execute(accelerated_time):

    trajectory_index = 49
    print("using ball trajectory file: ",
          context.BallTrajectories().get_file_name(trajectory_index))
    
    hysr = HysrOneBall(accelerated_time,
                       o80_pam_time_step,
                       mujoco_time_step,
                       algo_time_step,
                       None, # no reference posture to go to between episodes
                       target_position,
                       reward_normalization_constant,
                       smash_task,
                       rtt_cap=rtt_cap,
                       trajectory_index=trajectory_index) # always plays this trajectory. Set to None for random.

    hysr.reset()
    frequency_manager = o80.FrequencyManager(1.0/algo_time_step)

    # time_switch: duration after episode start after which
    # the robot performs the swing motion
    # manually selected so that sometimes the racket hits the ball,
    # sometimes it does not
    ts = 0.5
    time_switches = []
    while ts<=0.8:
        time_switches.append(ts)
        ts+=0.025

    # converting time switches from seconds to nb of iterations
    iteration_switches = [(1.0/o80_pam_time_step) * ts for ts in time_switches]

    for episode,iteration_switch in enumerate(iteration_switches):

        print("EPISODE",episode,iteration_switch)

        start_robot_iteration = hysr.get_robot_iteration()
        start_ball_iteration = hysr.get_ball_iteration()
        
        running = True

        while running:

            current_iteration = hysr.get_robot_iteration()

            if  (current_iteration - start_robot_iteration) < iteration_switch :
                pressures = reference_posture
            else:
                pressures = swing_posture

            observation,reward,reset = hysr.step(pressures)

            if not accelerated_time:
                waited = frequency_manager.wait()
                if waited<0:
                    print("! warning ! failed to maintain algorithm frequency")

            if reset:
                print("\treward:",reward)

            running = not reset


        hysr.reset()
        frequency_manager = o80.FrequencyManager(1.0/algo_time_step)
        
    hysr.close()

def _configure():
    config = BrightArgs(str("hysr dummy demo using swing motion.\n"+
                            "to be started after start_robots or start_robots_accelerated.\n"+
                            "(in same folder)"))
    config.add_operation("accelerated",
                         "if used, start_robot_accelerated must have been started.")
    change_all=False
    finished  = config.dialog(change_all,sys.argv[1:])
    if not finished:
        return None
    return config


def _run():
    config = _configure()
    if config is None:
        return
    execute(config.accelerated)
    

if __name__ == "__main__":
    _run()
