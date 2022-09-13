import pathlib
import json
import os
import site
import sys
import time

import o80
import o80_pam
import pam_interface
import pam_mujoco
import context
import frequency_monitoring
import shared_memory
from pam_mujoco import mirroring
from my_pd_controller import MyPositionController
from learning_table_tennis_from_scratch import configure_mujoco
from learning_table_tennis_from_scratch import robot_integrity
import numpy as np
import math

SEGMENT_ID_BALL = pam_mujoco.segment_ids.ball
SEGMENT_ID_GOAL = pam_mujoco.segment_ids.goal
SEGMENT_ID_HIT_POINT = pam_mujoco.segment_ids.hit_point
SEGMENT_ID_ROBOT_MIRROR = pam_mujoco.segment_ids.mirroring
SEGMENT_ID_PSEUDO_REAL_ROBOT = o80_pam.segment_ids.robot
SEGMENT_ID_EPISODE_FREQUENCY = "hysr_episode_frequency"
SEGMENT_ID_STEP_FREQUENCY = "hysr_step_frequency"


def _to_robot_type(robot_type: str) -> pam_mujoco.RobotType:
    try:
        return pam_mujoco.RobotType[robot_type.upper()]
    except KeyError:
        error = str(
            "hysr configuration robot_type should be either "
            "'pamy1' or 'pamy2' (entered value: {})"
        ).format(robot_type)
        raise ValueError(error)


class HysrOneBallConfig:

    __slots__ = (
        "real_robot",
        "robot_type",
        "o80_pam_time_step",
        "mujoco_time_step",
        "algo_time_step",
        "pam_config_file",
        "robot_position",
        "robot_orientation",
        "table_position",
        "table_orientation",
        "target_position",
        "reference_posture",
        "starting_pressures",
        "world_boundaries",
        "pressure_change_range",
        "trajectory",
        "accelerated_time",
        "graphics_pseudo_real",
        "graphics_simulation",
        "graphics_extra_balls",
        "instant_reset",
        "nb_steps_per_episode",
        "extra_balls_sets",
        "extra_balls_per_set",
        "trajectory_group",
        "frequency_monitoring_step",
        "frequency_monitoring_episode",
        "robot_integrity_check",
        "robot_integrity_threshold",
    )

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)

    def get(self):
        r = {s: getattr(self, s) for s in self.__slots__}
        return r

    @classmethod
    def from_json(cls, jsonpath):
        if not os.path.isfile(jsonpath):
            raise FileNotFoundError(
                "failed to find hysr configuration file: {}".format(jsonpath)
            )
        try:
            with open(jsonpath) as f:
                conf = json.load(f)
        except Exception as e:
            raise ValueError(
                "failed to parse reward json configuration file {}: {}".format(
                    jsonpath, e
                )
            )
        instance = cls()
        for s in cls.__slots__:
            try:
                setattr(instance, s, conf[s])
            except Exception:
                raise ValueError(
                    "failed to find the attribute {} " "in {}".format(s, jsonpath)
                )
        # robot type given as string in json config, but
        # the rest of the code will expect a pam_mujoco.RobotType
        instance.robot_type = _to_robot_type(instance.robot_type)

        # convert paths to Path objects and expand '~'
        instance.pam_config_file = pathlib.Path(instance.pam_config_file).expanduser()

        return instance

    @staticmethod
    def default_path():
        global_install = os.path.join(
            sys.prefix,
            "local",
            "learning_table_tennis_from_scratch_config",
            "hysr_one_ball_default.json",
        )
        local_install = os.path.join(
            site.USER_BASE,
            "learning_table_tennis_from_scratch_config",
            "hysr_one_ball_default.json",
        )

        if os.path.isfile(local_install):
            return local_install
        if os.path.isfile(global_install):
            return global_install


    
def _convert_pressures_in(pressures):
    # convert pressure from [ago1, antago1, ago2, antago2, ...]
    # to [(ago1, antago1), (ago2, antago2), ...]
    return list(zip(pressures[::2], pressures[1::2]))


def _convert_pressures_out(pressures_ago, pressures_antago):
    pressures = list(zip(pressures_ago, pressures_antago))
    return [p for sublist in pressures for p in sublist]


class _Observation:
    """
    Needs : jont_pos, joint_vel, pressures_both, pressure_diff, pos_diff, vel_diff, pd_command
    """
    def __init__(
        self,
        joint_positions,
        joint_velocities,
        pressures,
        pressure_diff,
        joint_pos_diff,
        joint_vel_diff,
        pd_command

    ):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.pressures = pressures
        self.pressure_diff = pressure_diff
        self.joint_pos_diff = joint_pos_diff
        self.joint_vel_diff = joint_vel_diff
        self.pd_command = pd_command

    def flatten(self):
        obs = (self.joint_positions +
            self.joint_velocities+
            self.pressures+
            self.pressure_diff+
            self.joint_pos_diff+
            self.joint_vel_diff+
            self.pd_command
        )
        return obs


class HysrOneBall_single_robot:
    def __init__(self, hysr_config, reward_function, logs=False):

        self._hysr_config = hysr_config
        self.logs = logs
        self.logger = []

        # we will track the episode number
        self._episode_number = -1

        # we will track the step number (reset at the start
        # of each episode)
        self._step_number = -1

        # we end an episode after a fixed number of steps
        self._nb_steps_per_episode = hysr_config.nb_steps_per_episode
        # note: if self._nb_steps_per_episode is 0 or less,
        #       an episode will end based on a threshold
        #       in the z component of the ball position
        #       (see method _episode_over)

        # this instance of HysrOneBall interacts with several
        # instances of mujoco (pseudo real robot, simulated robot,
        # possibly instances of mujoco for extra balls).
        # Listing all the corresponding mujoco_ids
        self._mujoco_ids = []

        # pam muscles configuration
        self._pam_config = pam_interface.JsonConfiguration(
            str(hysr_config.pam_config_file)
        )

        # to control pseudo-real robot (pressure control)
        if not hysr_config.real_robot:
            (
                self._real_robot_handle,
                self._real_robot_frontend,
            ) = configure_mujoco.configure_pseudo_real(
                str(hysr_config.pam_config_file),
                hysr_config.robot_type,
                graphics=hysr_config.graphics_pseudo_real,
                accelerated_time=hysr_config.accelerated_time,
            )
            self._mujoco_ids.append(self._real_robot_handle.get_mujoco_id())
        else:
            # real robot: making some sanity check that the
            # rest of the configuration is ok
            if hysr_config.instant_reset:
                raise ValueError(
                    str(
                        "HysrOneBall configured for "
                        "real robot and instant reset."
                        "Real robot does not support "
                        "instant reset."
                    )
                )
            if hysr_config.accelerated_time:
                raise ValueError(
                    str(
                        "HysrOneBall configured for "
                        "real robot and accelerated time."
                        "Real robot does not support "
                        "accelerated time."
                    )
                )

        # # to control the simulated robot (joint control)
        # self._simulated_robot_handle = configure_mujoco.configure_simulation(
        #     hysr_config
        # )
        # self._mujoco_ids.append(self._simulated_robot_handle.get_mujoco_id())

        # # where we want to shoot the ball
        # self._target_position = hysr_config.target_position
        # self._goal = self._simulated_robot_handle.interfaces[SEGMENT_ID_GOAL]

        # # to read all recorded trajectory files
        # self._trajectory_reader = context.BallTrajectories(hysr_config.trajectory_group)
        # _BallBehavior.read_trajectories(hysr_config.trajectory_group)

        # if requested, logging info about the frequencies of the steps and/or the
        # episodes
        if hysr_config.frequency_monitoring_step:
            size = 1000
            self._frequency_monitoring_step = frequency_monitoring.FrequencyMonitoring(
                SEGMENT_ID_STEP_FREQUENCY, size
            )
        else:
            self._frequency_monitoring_step = None
        if hysr_config.frequency_monitoring_episode:
            size = 1000
            self._frequency_monitoring_episode = (
                frequency_monitoring.FrequencyMonitoring(
                    SEGMENT_ID_EPISODE_FREQUENCY, size
                )
            )
        else:
            self._frequency_monitoring_episode = None

        # if o80_pam (i.e. the pseudo real robot)
        # has been started in accelerated time,
        # the corresponding o80 backend will burst through
        # an algorithm time step
        self._accelerated_time = hysr_config.accelerated_time
        if self._accelerated_time:
            self._o80_time_step = hysr_config.o80_pam_time_step
            self._nb_robot_bursts = int(
                hysr_config.algo_time_step / hysr_config.o80_pam_time_step
            )

        # pam_mujoco (i.e. simulated ball and robot) should have been
        # started in accelerated time. It burst through algorithm
        # time steps
        self._mujoco_time_step = hysr_config.mujoco_time_step
        self._nb_sim_bursts = int(
            hysr_config.algo_time_step / hysr_config.mujoco_time_step
        )

        # the config sets either a zero or positive int (playing the
        # corresponding indexed pre-recorded trajectory) or a negative int
        # (playing randomly selected indexed trajectories)
        # if hysr_config.trajectory >= 0:
        #     self._ball_behavior = _BallBehavior(index=hysr_config.trajectory)
        # else:
        #     self._ball_behavior = _BallBehavior(random=True)

        # the robot will interpolate between current and
        # target posture over this duration
        self._period_ms = hysr_config.algo_time_step

        # reward configuration
        self._reward_function = reward_function

        # to get information regarding the ball
        # (instance of o80_pam.o80_ball.o80Ball)
        # self._ball_communication = self._simulated_robot_handle.interfaces[
        #     SEGMENT_ID_BALL
        # ]

        # to send pressure commands to the real or pseudo-real robot
        # (instance of o80_pam.o80_pressures.o80Pressures)
        # hysr_config.real robot is either false (i.e. pseudo real
        # mujoco robot) or the segment_id of the real robot backend
        if not hysr_config.real_robot:
            self._pressure_commands = self._real_robot_handle.interfaces[
                SEGMENT_ID_PSEUDO_REAL_ROBOT
            ]
        else:
            self._real_robot_frontend = o80_pam.FrontEnd(hysr_config.real_robot)
            self._pressure_commands = o80_pam.o80Pressures(
                hysr_config.real_robot, frontend=self._real_robot_frontend
            )

        # will encapsulate all information
        # about the ball (e.g. min distance with racket, etc)
        # self._ball_status = context.BallStatus(hysr_config.target_position)

        # to send mirroring commands to simulated robots
        # self._mirrorings = [
        #     self._simulated_robot_handle.interfaces[SEGMENT_ID_ROBOT_MIRROR]
        # ]

        # to move the hit point marker
        # (instance of o80_pam.o80_hit_point.o80HitPoint)
        # self._hit_point = self._simulated_robot_handle.interfaces[SEGMENT_ID_HIT_POINT]

        # tracking if this is the first step of the episode
        # (a call to the step function sets it to false, call to reset function sets it
        # back to true)
        self._first_episode_step = True

        # normally an episode ends when the ball z position goes
        # below a certain threshold (see method _episode_over)
        # this is to allow user to force ending an episode
        # (see force_episode_over method)
        self._force_episode_over = False

        # if false, the system will reset via execution of commands
        # if true, the system will reset by resetting the simulations
        # Only "false" is supported by the real robot
        self._instant_reset = hysr_config.instant_reset

        # adding extra balls (if any)
        # if (
        #     hysr_config.extra_balls_sets is not None
        #     and hysr_config.extra_balls_sets > 0
        # ):

        #     self._extra_balls = []

        #     for setid in range(hysr_config.extra_balls_sets):

        #         # balls: list of instances of _ExtraBalls (defined in this file)
        #         # mirroring : for sending mirroring command to the robot
        #         #             of the set (joint controlled)
        #         #             (instance of
        #         #             o80_pam.o80_robot_mirroring.o80RobotMirroring)
        #         balls, mirroring, mujoco_id, frontend = _get_extra_balls(
        #             setid, hysr_config
        #         )

        #         self._extra_balls.extend(balls)
        #         self._mirrorings.append(mirroring)
        #         self._mujoco_ids.append(mujoco_id)
        #         self._extra_balls_frontend = frontend
        # else:
        #     self._extra_balls = []
        #     self._extra_balls_frontend = None

        # # for running all simulations (main + for extra balls)
        # # in parallel (i.e. when bursting is called, all mujoco
        # # instance perform step(s) in parallel)
        # self._parallel_burst = pam_mujoco.mirroring.ParallelBurst(self._mirrorings)

        # if set, logging the position of the robot at the end of reset, and possibly
        # get a warning when this position drifts as the number of episodes increase
        if hysr_config.robot_integrity_check is not None:
            self._robot_integrity = robot_integrity.RobotIntegrity(
                hysr_config.robot_integrity_threshold,
                file_path=hysr_config.robot_integrity_check,
            )
        else:
            self._robot_integrity = None

        # when starting, the real robot and the virtual robot(s)
        # may not be aligned, which may result in graphical issues,
        # so aligning them
        # (get values of real robot via self._pressure_commands,
        # and set values to all simulated robot via self._mirrorings)
        # source of mirroring in pam_mujoco.mirroring.py
        # pam_mujoco.mirroring.align_robots(self._pressure_commands, self._mirrorings)
        self.observatin = None
        self.last_observation = None
        self.controller = None
        self.frequency_manager = None
        self.command = None

    def get_starting_pressures(self):
        return self._hysr_config.starting_pressures

    def _share_episode_number(self, episode_number):
        # write the episode number in a memory shared
        # with the instances of mujoco
        for mujoco_id in self._mujoco_ids:
            shared_memory.set_long_int(mujoco_id, "episode", episode_number)

    def force_episode_over(self):
        # will trigger the method _episode_over
        # (called in the step method) to return True
        self._force_episode_over = True

    # def set_ball_behavior(self, line=False, index=False, random=False):
    #     # overwrite the ball behavior (set to a trajectory in the constructor)
    #     # see comments in _BallBehavior, in this file
    #     self._ball_behavior = _BallBehavior(line=line, index=index, random=random)

    # def set_extra_ball_behavior(
    #     self, ball_index, line=False, index=False, random=False
    # ):
    #     # overwrite the ball behavior of the extra ball (set to random
    #     # selected pre-recorded trajectory in constructor)
    #     # see comments in _BallBehavior, in this file
    #     if ball_index < 0 or ball_index >= len(self._extra_balls):
    #         raise IndexError(ball_index)
    #     self._extra_balls[ball_index].ball_behavior = _BallBehavior(
    #         line=line, index=index, random=random
    #     )

    # def _create_observation(self):
    #     (
    #         pressures_ago,
    #         pressures_antago,
    #         joint_positions,
    #         joint_velocities,
    #     ) = self._pressure_commands.read()
    #     # ball_position, ball_velocity = self._ball_communication.get()

    #     self.des_q = self.q_trajectory[self._step_number]
    #     self.des_dq = self.dq_trajectory[self._step_number]

    #     observation = _Observation(
    #         joint_positions,
    #         joint_velocities,
    #         _convert_pressures_out(pressures_ago, pressures_antago),
    #         pressures_ago - pressures_antago,
    #         self.des_q - joint_positions,
    #         self.des_dq - joint_velocities,
    #         #PD action

    #     )
    #     return observation.flatten()

    def get_robot_iteration(self):
        return self._pressure_commands.get_iteration()

    # def get_ball_iteration(self):
    #     return self._ball_communication.get_iteration()

    def get_current_desired_pressures(self):
        (pressures_ago, pressures_antago, _, __) = self._pressure_commands.read(
            desired=True
        )
        return pressures_ago, pressures_antago

    def get_current_pressures(self):
        (pressures_ago, pressures_antago, _, __) = self._pressure_commands.read(
            desired=False
        )
        return pressures_ago, pressures_antago

    def _do_natural_reset(self):
        self._move_to_position(self._hysr_config.reference_posture)

    def _do_instant_reset(self):

        # "instant": reset all mujoco instances
        # to their starting state. Not applicable
        # to real robot

        self._real_robot_handle.reset()
        # self._simulated_robot_handle.reset()
        # for handle in _ExtraBall.handles.values():
        #     handle.reset()
        self._move_to_pressure(self._hysr_config.reference_posture)

    def _move_to_pressure(self, pressures):
        # moves to pseudo-real robot to desired pressure in synchronization
        # with the simulated robot(s)
        if self._accelerated_time:
            for _ in range(self._nb_robot_bursts):
                self._pressure_commands.set(pressures, burst=1)
                _, _, joint_positions, joint_velocities = self._pressure_commands.read()
                # for mirroring_ in self._mirrorings:
                #     mirroring_.set(joint_positions, joint_velocities)
                # self._parallel_burst.burst(1)
            return
        else:
            self._pressure_commands.set(pressures, burst=False)
        time_start = self._real_robot_frontend.latest().get_time_stamp() * 1e-9
        current_time = time_start
        timeout = 0.5
        while current_time - time_start < timeout:
            current_time = self._real_robot_frontend.latest().get_time_stamp() * 1e-9
            _, _, joint_positions, joint_velocities = self._pressure_commands.read()
            # for mirroring_ in self._mirrorings:
            #     mirroring_.set(joint_positions, joint_velocities)
            # self._parallel_burst.burst(self._nb_sim_bursts)

    def _move_to_position(self, position):
        # moves the pseudo-real robot to a desired position (in radians)
        # via a position controller (i.e. compute the pressure trajectory
        # required to reach, hopefully, for the position) in
        # synchronization with the simulated robot(s).

        # configuration for the controller
        KP = [0.8, -3.0, 1.2, -1.0]
        KI = [0.015, -0.25, 0.02, -0.05]
        KD = [0.04, -0.09, 0.09, -0.09]
        NDP = [-0.3, -0.5, -0.34, -0.48]
        TIME_STEP = 0.01  # seconds
        QD_DESIRED = [0.7, 0.7, 0.7, 0.7]  # radian per seconds
        _, _, Q_CURRENT, _ = self._pressure_commands.read()

        # configuration for HYSR
        NB_SIM_BURSTS = int((TIME_STEP / self._hysr_config.mujoco_time_step) + 0.5)

        # configuration for accelerated time
        if self._accelerated_time:
            NB_ROBOT_BURSTS = int((TIME_STEP / hysr_config.o80_pam_time_step) + 0.5)

        # configuration for real time
        if not self._accelerated_time:
            frequency_manager = o80.FrequencyManager(1.0 / TIME_STEP)

        # applying the controller twice yields better results
        for _ in range(2):

            _, _, q_current, _ = self._pressure_commands.read()

            # the position controller
            controller = o80_pam.PositionController(
                q_current,
                position,
                QD_DESIRED,
                self._pam_config,
                KP,
                KD,
                KI,
                NDP,
                TIME_STEP,
            )

            # rolling the controller
            while controller.has_next():
                # current position and velocity of the real robot
                _, _, q, qd = self._pressure_commands.read()
                # mirroing the simulated robot(s)
                # for mirroring_ in self._mirrorings:
                #     mirroring_.set(q, qd)
                # self._parallel_burst.burst(NB_SIM_BURSTS)
                # applying the controller to get the pressure to set
                pressures = controller.next(q, qd)
                # setting the pressures to real robot
                if self._accelerated_time:
                    # if accelerated times, running the pseudo real robot iterations
                    # (note : o80_pam expected to have started in bursting mode)
                    self._pressure_commands.set(pressures, burst=NB_ROBOT_BURSTS)
                else:
                    # Should start acting now in the background if not accelerated time
                    self._pressure_commands.set(pressures, burst=False)
                    frequency_manager.wait()

    def reset(self):

        
        # what happens during reset does not correspond
        # to any episode (-1 means: no active episode)
        self._share_episode_number(-1)

        # resetting the measure of step frequency monitoring
        if self._frequency_monitoring_step:
            self._frequency_monitoring_step.reset()

        # exporting episode frequency
        if self._frequency_monitoring_episode:
            self._frequency_monitoring_episode.ping()
            self._frequency_monitoring_episode.share()

        # in case the episode was forced to end by the
        # user (see force_episode_over method)
        self._force_episode_over = False

        # resetting first episode step
        self._first_episode_step = True

        # resetting the hit point
        # self._hit_point.set([0, 0, -0.62], [0, 0, 0])

        if self._instant_reset:
            # going back to vertical position
            # on simulated robot
            self._do_instant_reset()
            # mirroring.align_robots(self._pressure_commands, self._mirrorings)
        else:
            # moving to reset position
            self._do_natural_reset()

        # going to starting pressure
        self._move_to_pressure(self._hysr_config.starting_pressures)

        # moving the goal to the target position
        # self._goal.set(self._target_position, [0, 0, 0])

        # setting the ball behavior
        # self.load_ball()

        # control post contact was lost, restoring it
        # self._simulated_robot_handle.reset_contact(SEGMENT_ID_BALL)
        # self._simulated_robot_handle.deactivate_contact(SEGMENT_ID_BALL)
        # for ball in self._extra_balls:
        #     ball.handle.reset_contact(ball.segment_id)
        #     ball.handle.deactivate_contact(ball.segment_id)

        # moving the ball(s) to initial position
        # self._parallel_burst.burst(4)

        # resetting ball/robot contact information
        # self._simulated_robot_handle.reset_contact(SEGMENT_ID_BALL)
        # self._simulated_robot_handle.activate_contact(SEGMENT_ID_BALL)
        # for ball in self._extra_balls:
        #     ball.handle.reset_contact(ball.segment_id)
        #     ball.handle.activate_contact(ball.segment_id)

        time.sleep(0.1)
        # resetting ball info, e.g. min distance ball/racket, etc
        # self._ball_status.reset()
        # for ball in self._extra_balls:
        #     ball.ball_status.reset()

        # checking the position of the robot, to see if it drifts
        # as episode increase (or if it not what is expected at all).
        # raise an exception if drifted too much).
        if self._robot_integrity is not None:
            _, _, joint_positions, _ = self._pressure_commands.read()
            warning = self._robot_integrity.set(joint_positions)
            if warning:
                self._robot_integrity.close()
                self.close()
                raise robot_integrity.RobotIntegrityException(
                    self._robot_integrity, joint_positions
                )

        # a new episode starts
        self.reset_logger()
        self._step_number = 0
        self._episode_number += 1
        self._share_episode_number(self._episode_number)
        
        # Setting a target point and setting up the controller
        (
            pressures_ago,
            pressures_antago,
            joint_positions,
            joint_velocities,
        ) = self._pressure_commands.read()

        # SET TARGETS
        # desired positions
        pi4 = math.pi/4.0
        pi6 = math.pi/6.0
        q_current = joint_positions
        q_target = [-pi4,-pi6,+pi6,+pi6]
        qd_desired = [0.5] *4 # radian per seconds

        # the position controller
        self.create_position_controller(q_current, qd_desired, q_target)

        # Get the new observation
        (
            pressures_ago,
            pressures_antago,
            joint_positions,
            joint_velocities,
        ) = self._pressure_commands.read()

        next_pid_command = self.controller.peek_next_command(joint_positions, joint_velocities, step_mod = 0) #operates at t+1 since last next_command updated it 
        q_des, qd_des = self.controller.peek_next_desired_values()
        # observation instance
        observation = _Observation(
            joint_positions,
            joint_velocities,
            _convert_pressures_out(pressures_ago, pressures_antago),
            [ag-at for ag,at in zip(pressures_ago, pressures_antago)],
            [des - q for des,q in zip(q_des, joint_positions)],
            [des - qd for des,qd in zip(qd_des, joint_velocities)],
            next_pid_command,
        )

        self.observation = observation

        # #logging
        # logs = {
        #     'obs_next':observation,
        #     'obs_des':q_des.copy(),
        #     'obs_vel_des':qd_des.copy(),
        #     'action':0,
        #     'pid_command':pid_command.copy(),
        #     'full_command':command.copy(),
        #     'pressure_command': pressures.copy(),
        #     'reward':reward,
        #     'episode_over': episode_over
        # }

        # self.log_data(logs)
        # returning an observation
        return observation.flatten()

    def create_position_controller(self, q_current, qd_desired, q_target):
        # configuration for the controller
        # KP = [0.8, -3.0, 1.2, -1.0]
        # KI = [0.015, -0.25, 0.02, -0.05]
        # KD = [0.04, -0.09, 0.09, -0.09]
        # TIME_STEP = 0.01

        #tuned
        KP = [0.2,0.2,0.2,0.2]
        KD = [0.02,0.02,0.02,0.02]
        KI = [0.05,0.05,0.05,0.05]
        NDP = [0.5,0.5,0.6,0.5]
        TIME_STEP = self._hysr_config.algo_time_step
        # _, _, Q_CURRENT, _ = self._pressure_commands.read()

        # configuration for HYSR
        NB_SIM_BURSTS = int((TIME_STEP / self._hysr_config.mujoco_time_step) + 0.5)

        # configuration for accelerated time
        if self._accelerated_time:
            NB_ROBOT_BURSTS = int((TIME_STEP / hysr_config.o80_pam_time_step) + 0.5)

        # configuration for real time
        if not self._accelerated_time:
            self.frequency_manager = o80.FrequencyManager(1.0 / TIME_STEP)

        self.controller = MyPositionController(
            q_current,
            q_target,
            self._nb_steps_per_episode,
            self._pam_config,
            KP,
            KD,
            KI,
            NDP,
            TIME_STEP,
        )

        self.q_trajectory = self.controller._q_trajectories
        self.dq_trajectory = self.controller._dq_trajectories

    def _episode_over(self):
        
        # manual condition
        # if self._step_number > 500:
        #     return True

        # if self._nb_steps_per_episode is positive,
        # exiting based on the number of steps
        if self._nb_steps_per_episode > 0:
            if self._step_number >= self._nb_steps_per_episode:
                return True
            else:
                return False

        # otherwise exiting based on a threshold on the
        # z position of the ball

        # ball falled below the table
        # note : all prerecorded trajectories are added a last ball position
        # with z = -10.0, to insure this always occurs.
        # see: function reset
        # if self._ball_status.ball_position[2] < 0.8:
        #     return True

        # in case the user called the method
        # force_episode_over
        if self._force_episode_over:
            return True

        return False

    # def get_ball_position(self):
    #     # returning current ball position
    #     ball_position, _ = self._ball_communication.get()
    #     return ball_position

    # action assumed to be np.array(ago1,antago1,ago2,antago2,...)
    def step(self, action):

        # reading current real (or pseudo real) robot state
        # THIS should not be happening here. Because we need the PD component to calculate the action.
        # So lets understanding waiting a bit more.... Can we call get create observation before or should we just utilize the past observation
        # Commented out because the last observed obs should be used. it taken after waiting
        # (
        #     pressures_ago,
        #     pressures_antago,
        #     joint_positions,
        #     joint_velocities,
        # ) = self._pressure_commands.read()

        # convert action [ago1,antago1,ago2] to list suitable for
        # o80 ([(ago1,antago1),(),...])
        # pressures = _convert_pressures_in(list(action))

        #getting action in command values
        command_policy = action
        
        ## Traj Following :
        # 1. Get the PD controller value in torque command
        # 2. Add the action value to the torque value (appropriate range)
        # 3. Convert all to pressure using hueristic

        #1 .  get PD value
        if self.controller.has_next():
            # current position and velocity of the real robot - check here as this can be different, it should not be reading again
            # _, _, q, qd = self._pressure_commands.read()
            q = self.observation.joint_positions
            qd = self.observation.joint_velocities
            # applying the controller to get the pressure to set
            # pd_pressures = self.controller.next(q, qd)
            pid_command = self.controller.next_command(q, qd, step_mod = 0 ) #-1 because step was updated in next command before (TEST ONLY)
            # pd_pressures2 = self.controller.convert_command_to_pressures(command)
        else:
            print("Controller reached position, do something HERE") #what happens when there is nothing left

        #2. Add pressures/torques
        command = [c+pi for c,pi in zip(pid_command, command_policy)]

        #3. Convert torques to pressure using heuristic
        pressures = self.controller.convert_command_to_pressures(command)

        # sending action pressures to real (or pseudo real) robot.
        # Waiting needs to be HERE, since then we can get the updated observation.
        # Being outside causes the observation to become old
        # then we can even eliminate the new observation pull at the start

        if self._accelerated_time:
            # if accelerated times, running the pseudo real robot iterations
            # (note : o80_pam expected to have started in bursting mode)
            self._pressure_commands.set(pressures, burst=self._nb_robot_bursts)
        else:
            # Should start acting now in the background if not accelerated time
            self._pressure_commands.set(pressures, burst=False)
            waited = self.frequency_manager.wait() # adding here
            print("Waited for : ",waited)
            if waited<0:
                print("Failed to meet algo frequency")

        # # sending mirroring state to simulated robot(s)
        # for mirroring_ in self._mirrorings:
        #     mirroring_.set(joint_positions, joint_velocities)

        # # having the simulated robot(s)/ball(s) performing the right number of
        # # iterations (note: simulated expected to run accelerated time)
        # self._parallel_burst.burst(self._nb_sim_bursts)

        # def _update_ball_status(handle, segment_id, ball_status):
        #     # getting ball/racket contact information
        #     # note : racket_contact_information is an instance
        #     #        of context.ContactInformation
        #     racket_contact_information = handle.get_contact(segment_id)
        #     # updating ball status
        #     ball_status.update(ball_position, ball_velocity, racket_contact_information)

        # # updating the status of all balls
        # _update_ball_status(
        #     self._simulated_robot_handle, SEGMENT_ID_BALL, self._ball_status
        # )
        # for ball in self._extra_balls:
        #     _update_ball_status(ball.handle, ball.segment_id, ball.ball_status)

        # # moving the hit point to the minimal observed distance
        # # between ball and target (post racket hit)
        # if self._ball_status.min_position_ball_target is not None:
        #     self._hit_point.set(self._ball_status.min_position_ball_target, [0, 0, 0]

        # Get the new observation
        (
            pressures_ago,
            pressures_antago,
            joint_positions,
            joint_velocities,
        ) = self._pressure_commands.read()

        #get reward
        reward = self.get_reward(joint_positions, joint_velocities,command)

        q = joint_positions
        qd = joint_velocities
        next_pid_command = self.controller.peek_next_command(q, qd, step_mod = 0) #operates at t+1 since last next_command updated it 

        q_des, qd_des = self.controller.peek_next_desired_values()

        # observation instance
        observation = _Observation(
            joint_positions,
            joint_velocities,
            _convert_pressures_out(pressures_ago, pressures_antago),
            [ag-at for ag,at in zip(pressures_ago, pressures_antago)],
            [des - q for des,q in zip(q_des,joint_positions)],
            [des - qd for des,qd in zip(qd_des, joint_velocities)],
            next_pid_command,
        )

        self.observation = observation
        # checking if episode is over
        episode_over = self._episode_over()

        # next step can not be the first one
        # (reset will set this back to True)
        self._first_episode_step = False

        # exporting step frequency ---- WHAT IS THIS ? ANYTHING TO DO WITH FREQ MANAGER?
        if self._frequency_monitoring_step:
            self._frequency_monitoring_step.ping()
            self._frequency_monitoring_step.share()

        # this step is done
        self._step_number += 1

        #logging
        logs = {
            'obs_next':observation.flatten().copy(),
            'obs_des':q_des.copy(),
            'obs_vel_des':qd_des.copy(),
            'action':command_policy.copy(),
            'pid_command':pid_command.copy(),
            'full_command':command.copy(),
            'pressure_command': pressures.copy(),
            'reward':reward,
            'episode_over': episode_over
        }

        if self.logs:
            self.log_data(logs)

        # returning
        return observation.flatten(), reward, episode_over, logs

    def close(self):
        if self._robot_integrity is not None:
            self._robot_integrity.close()
        # self._parallel_burst.stop()
        shared_memory.clear_shared_memory(SEGMENT_ID_EPISODE_FREQUENCY)
        shared_memory.clear_shared_memory(SEGMENT_ID_STEP_FREQUENCY)
        pass

    def get_reward(self, joint_positions, joint_velocities, command):
        last_velocities = np.array(self.observation.joint_velocities)
        joint_velocities = np.array(joint_velocities)
        joint_acc = (joint_velocities - last_velocities)/ self._hysr_config.algo_time_step

        command_acc = np.array(command)
        reward = -1*np.linalg.norm(joint_acc - command_acc)
        return reward


    def log_data(self, logs):
        self.logger.append(logs)
    

    def dump_logger(self):
        joint_pos = []
        joint_vel= []
        actions = []
        rewards = []
        pid_command = []
        joint_pos_des = []
        joint_vel_des = []
        for items in self.logger:
            joint_pos.append(items['obs_next'][:4].copy())
            joint_vel.append(items['obs_next'][4:8].copy())
            actions.append(items['action'].copy())
            rewards.append(items['reward'])
            pid_command.append(items['pid_command'].copy())
            joint_pos_des.append(items['obs_des'].copy())
            joint_vel_des.append(items['obs_vel_des'].copy())

        all_data_np = {
        'joint_pos' : np.array(joint_pos),
        'joint_vel' : np.array(joint_vel),
        'actions' : np.array(actions),
        'rewards' : np.array(rewards),
        'pid_command' : np.array(pid_command),
        'joint_pos_des' : np.array(joint_pos_des),
        'joint_vel_des':np.array(joint_vel_des)
        }

        self.reset_logger()

        return all_data_np

    def reset_logger(self):
        self.logger= []

