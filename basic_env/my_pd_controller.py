import typing, copy
from typing import Sequence, Tuple
from dataclasses import dataclass
import math
import pam_interface
import numpy as np

@dataclass
class PositionControllerStep:
    step: int
    q: float
    dq: float
    error: float
    d_error: float
    q_traj: float
    dq_traj: float
    control_p: float
    control_d: float
    control_i: float
    control: float
    p_ago: int
    p_antago: int
    

class MyPositionController:
    """
    Computes a trajectory of pressures suitable to reach
    a desired posture (i.e. desired angular positions of
    joints). To compute the pressures: a PID controller is used
    to compute a control signal. The control signal is used to set
    up an ratio of agonist/antagonist pressures to apply.

    EDIT : returns the command, and has a separate function to convert commmand to pressure

    Args:
        q_current: current joint positions (in radian)
        q_desired: target joint positions (in radian)
        dq_desired: target joint speed during motion (in radian per second)
        pam_interface_config: configuration of the PAM muscles
        kp,kd,ki : PID gains
        ndp: pressure level gain. The higher the value, the higher the pressures
        time_step: the next function is expected to be called with
                    a period of time_step (seconds)
        extra_steps: the trajectory is extended of some extra steps which
                     have the final position with a velocity of 0 as
                     desired state. Helps the system to stabilize.
    """

    def __init__(
        self,
        q_current: Sequence[float],
        q_desired: Sequence[float],
        total_timesteps: int,
        pam_interface_config: pam_interface.Configuration,
        kp: Sequence[float],
        kd: Sequence[float],
        ki: Sequence[float],
        ndp: Sequence[float],
        time_step: float,
        extra_steps: int = 100,
        random_traj = False,
        test_traj = None,
        A_rng = None,
        B_rng = None
    ):

        self._q_current = q_current
        # self._q_desired = q_desired
        self._total_timesteps = total_timesteps
        # extra_steps = int(0.1*self._total_timesteps)
        extra_steps = 0
        self.extra_steps = extra_steps

        self._min_agos = pam_interface_config.min_pressures_ago
        self._max_agos = pam_interface_config.max_pressures_ago
        self._min_antagos = pam_interface_config.min_pressures_antago
        self._max_antagos = pam_interface_config.max_pressures_antago
        self._range_agos = [
            max_ - min_ for max_, min_ in zip(self._max_agos, self._min_agos)
        ]
        self._range_antagos = [
            max_ - min_ for max_, min_ in zip(self._max_antagos, self._min_antagos)
        ]

        self._kp = kp
        self._kd = kd
        self._ki = ki
        self._ndp = ndp

        self._time_step = time_step
        
        #traj
        self.random_traj = random_traj
        self.test_traj = test_traj
        
        self.A_rng = A_rng
        self.B_rng = B_rng
        self.traj_A_limits = [0.1, 0.7]
        self.traj_B_limits = [0.5, 3]
        
        self.steps = [self._total_timesteps for i in range(len(self._q_current))]

        # with desired q target
        # q_error = [
        #     current - desired
        #     for desired, current in zip(self._q_current, self._q_desired)
        # ]
        # steps = [
        #     math.ceil(abs(error) / time_step / dq)
        #     for error, dq in zip(q_error, self._dq_desired)
        # ]
        # steps = [self._total_timesteps for i in range(len(q_error))]
        # self._dq_desired = [er/ts for er,ts in zip(q_error, steps)]

        # def _get_q_trajectory(nb_steps, current, desired):
        #     error = desired - current
        #     r = [(error / nb_steps) * step + current for step in range(nb_steps)]
        #     return r
            
        # q_trajectories = [
        #     _get_q_trajectory(nb_steps, current, desired)
        #     for nb_steps, current, desired in zip(
        #         steps, self._q_current, self._q_desired
        #     )
        # ]
        # dq_trajectories = [
        #     [d_desired] * nb_steps
        #     for d_desired, nb_steps in zip(self._dq_desired, steps)
        # ]
        
        # SIN wave
        # A = 1
        # B = 2
        # q_trajectories = [[A*np.sin((B*np.pi/step) *s)+current for s in range(step)] for step,current in zip(self.steps,self._q_current)]
        # dq_trajectories = [[A*(B*np.pi/step)*np.cos((B*np.pi/step)*s) for s in range(step)]  for step in self.steps]
        # ddq_trajectories = [[-A*((B*np.pi/step)**2)*np.sin((B*np.pi/step)*s) for s in range(step)] for step in self.steps]
        

        # def _align_sizes(arrays, fill_values):
        #     max_size = max([len(a) for a in arrays]) + extra_steps

        #     def _align_size(array, fill_value):
        #         array.extend([fill_value] * (max_size - len(array)))
        #         return array

        #     return list(map(_align_size, arrays, fill_values))

        # #check if this is even required
        # self._q_trajectories = _align_sizes(q_trajectories, q_trajectories[-1])
        # self._dq_trajectories = _align_sizes(dq_trajectories, [0] * len(q_trajectories[-1]))
        # self._ddq_trajectories = _align_sizes(ddq_trajectories, [0] * len(q_trajectories[-1]))
        
        self.initialize_traj()
        
        self._error_sum = [0] * len(q_current)

        self._step = 0
        self._max_step = max(self.steps) + extra_steps

        self._introspect = [None]*len(self.steps)
        self._p_command = []
        self._i_command = []
        self._d_command = []
        self._pid_command = []
    
    
    def initialize_traj(self):
        if self.random_traj:
            # A = self.traj_A_limits[0] + self.A_rng.uniform()*(self.traj_A_limits[1] - self.traj_A_limits[0])
            # B = self.traj_B_limits[0] + self.B_rng.uniform()*(self.traj_B_limits[1] - self.traj_B_limits[0])
            A = self.traj_A_limits[0] + np.random.uniform()*(self.traj_A_limits[1] - self.traj_A_limits[0])
            B = self.traj_B_limits[0] + np.random.uniform()*(self.traj_B_limits[1] - self.traj_B_limits[0])
            
        elif self.test_traj == 'in_distribution':
            #think more on how to keep this uniform
            A = self.traj_A_limits[0] + self.A_rng.uniform()*(self.traj_A_limits[1] - self.traj_A_limits[0])
            B = self.traj_B_limits[0] + self.B_rng.uniform()*(self.traj_B_limits[1] - self.traj_B_limits[0])
            print("A = ", A , " | B = ",B)
        elif self.test_traj == 'out_of_distribution':
            a_limits = [0.7,1.1]
            b_limits = [3,4]
            A = a_limits[0] + self.A_rng.uniform()*(a_limits[1] - a_limits[0])
            B = b_limits[0] + self.B_rng.uniform()*(b_limits[1] - b_limits[0])
            print("A = ", A , " | B = ",B)
        else:
            # basic mode -- single traj -- same test and eval traj
            # SIN wave
            A = 1
            B = 2
        
        # print("A = ", A , " | B = ",B)
        q_trajectories = [[A*np.sin((B*np.pi/step) *s)+current for s in range(step)] for step,current in zip(self.steps,self._q_current)]
        dq_trajectories = [[A*(B*np.pi/step)*np.cos((B*np.pi/step)*s) for s in range(step)]  for step in self.steps]
        ddq_trajectories = [[-A*((B*np.pi/step)**2)*np.sin((B*np.pi/step)*s) for s in range(step)] for step in self.steps]

        def _align_sizes(arrays, fill_values):
            max_size = max([len(a) for a in arrays]) + self.extra_steps

            def _align_size(array, fill_value):
                array.extend([fill_value] * (max_size - len(array)))
                return array

            return list(map(_align_size, arrays, fill_values))

        #check if this is even required
        self._q_trajectories = _align_sizes(q_trajectories, q_trajectories[-1])
        self._dq_trajectories = _align_sizes(dq_trajectories, [0] * len(q_trajectories[-1]))
        self._ddq_trajectories = _align_sizes(ddq_trajectories, [0] * len(q_trajectories[-1]))
        
    
    # def reset_traj_seeds(self, traj_seed):
    #     print("Resetting Traj generator state")
    #     self.traj_seed = traj_seed
    #     self.A_rng = np.random.default_rng(seed = self.traj_seed[0])
    #     self.B_rng = np.random.default_rng(seed = self.traj_seed[1])
    
    def introspection(self)->typing.Sequence[PositionControllerStep]:
        return copy.deepcopy(self._introspect)
        
    def get_time_step(self)->float:
        return self._time_step
        
    def _next(self, dof: int, q: float, qd: float, step: int) -> Tuple[float, float]:
        error = self._q_trajectories[dof][step] - q
        d_error = self._dq_trajectories[dof][step] -qd
        self._error_sum[dof] += error
        control_p = self._kp[dof] * error
        control_d = self._kd[dof] * d_error
        control_i = self._ki[dof] * self._error_sum[dof]
        self._p_command = control_p
        self._i_command = control_i
        self._d_command = control_d
        control = control_p + control_d + control_i
        control = max(min(control, 1), -1)
        self._pid_command = control
        p_ago = self._min_agos[dof] + self._range_agos[dof] * (self._ndp[dof] - control)
        p_antago = self._min_antagos[dof] + self._range_antagos[dof] * (
            self._ndp[dof] + control
        )
        self._introspect[dof]=PositionControllerStep(
            step,q,qd,error,d_error,
            self._q_trajectories[dof][step],self._dq_trajectories[dof][step],
            control_p,control_d,control_i,control,
            int(p_ago),int(p_antago)
        )
        return int(p_ago), int(p_antago)

    def _next_command(self, dof: int, q: float, qd: float, step: int) -> Tuple[float, float]:
        error = self._q_trajectories[dof][step] - q
        d_error = self._dq_trajectories[dof][step] -qd
        self._error_sum[dof] += error
        control_p = self._kp[dof] * error
        control_d = self._kd[dof] * d_error
        control_i = self._ki[dof] * self._error_sum[dof]
        self._p_command = control_p
        self._i_command = control_i
        self._d_command = control_d
        control = control_p + control_d + control_i + self._ddq_trajectories[dof][step]
        control = max(min(control, 1), -1)
        self._pid_command = control
        return control

    def _next_command_peek(self, dof: int, q: float, qd: float, step: int) -> Tuple[float, float]:
        """
        Ensures class attributes are not updated as we are only peeking
        """
        error = self._q_trajectories[dof][step] - q
        d_error = self._dq_trajectories[dof][step] -qd
        _error_sum_dof = self._error_sum[dof] + error
        control_p = self._kp[dof] * error
        control_d = self._kd[dof] * d_error
        control_i = self._ki[dof] * _error_sum_dof
        self._p_command = control_p
        self._i_command = control_i
        self._d_command = control_d
        control = control_p + control_d + control_i + self._ddq_trajectories[dof][step]
        control = max(min(control, 1), -1)
        self._pid_command = control
        return control

    def has_next(self) -> bool:
        """
        Returns:
            False if the trajectory is finished, True otherwise
        """
        return self._step < self._max_step

    def next(self, q, qd) -> Sequence[Tuple[float, float]]:
        """
        Returns:
             a list [(pressure ago, pressure antago),...] of pressures to
             apply at the next step in order to follow the computed trajectory
        """
        r = list(map(self._next, range(len(q)), q, qd, [self._step] * len(q)))
        self._step += 1
        return r

    def next_command(self, q, qd, step_mod= 0):
        self._step = self._step + step_mod
        if self._step < self._max_step:
            r = list(map(self._next_command, range(len(q)), q, qd, [self._step] * len(q)))
        else: 
            r = list(map(self._next_command, range(len(q)), q, qd, [self._step-1] * len(q)))
        self._step = self._step + 1
        return r

    def convert_command_to_pressures(self, command):
        control = [max(min(c, 1), -1) for c in command]
        pressures = []
        for dof in range(len(control)):
            p_ago = self._min_agos[dof] + self._range_agos[dof] * (self._ndp[dof] - control[dof])
            p_antago = self._min_antagos[dof] + self._range_antagos[dof] * (
                self._ndp[dof] + control[dof]
            )
            pressures.append((int(p_ago), int(p_antago)))
        return pressures

    def peek_next_command(self, q, qd, step_mod= 0):
        """
        Gets next command without updating the _step
        """
        self._step = self._step + step_mod
        if self._step < self._max_step:
            r = list(map(self._next_command_peek, range(len(q)), q, qd, [self._step] * len(q)))
        else:
            # at last timestep, returning the command with prev desired values
            r = list(map(self._next_command_peek, range(len(q)), q, qd, [self._step-1] * len(q)))
        # self._step = self._step + 1
        return r        

    def peek_next_next_command(self, q, qd):
        next_step = min(self._step + 1, len(self._q_trajectories)-1)
        r = list(map(self._next_command, range(len(q)), q, qd, [next_step] * len(q)))
        return r

    def peek_next_desired_values(self):
        """
        Gets the desired values 
        """
        if self._step < self._max_step:
            q_des = [self._q_trajectories[i][self._step] for i in range(len(self._q_trajectories))]
            qd_des = [self._dq_trajectories[i][self._step] for i in range(len(self._dq_trajectories))]
        else:
            # at last timestep, returning the command with prev desired values
            q_des = [self._q_trajectories[i][self._step-1] for i in range(len(self._q_trajectories))]
            qd_des = [self._dq_trajectories[i][self._step-1] for i in range(len(self._dq_trajectories))]
        return q_des, qd_des