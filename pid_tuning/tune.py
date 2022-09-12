from ctypes import sizeof
from signal import pause
import socket, time
from struct import *
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from pathlib import Path
import os
from scipy.fft import fft, fftfreq


# ATTENTION
# install bayesOpt: pip install bayesian-optimization 
# downgrade scipy to version 1.7: pip install --upgrade scipy==1.7
# (see https://github.com/fmfn/BayesianOptimization/is  sues/300)


def setup_connection(serverAddressPort):
    # set up UDP connection
    serverAddressPort = ("192.168.0.110", 4700)
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    # send some message to trigger the robot to send messages
    init_msg = pack("=H",1)
    UDPClientSocket.sendto(init_msg, serverAddressPort)
    return UDPClientSocket
    
def read_pr_from_msg(UDPClientSocket,buffer_size):
    msgFromServer = UDPClientSocket.recvfrom(buffer_size)
    recMSG = unpack(
        "=LLLLLLLLLHHHHffffffffffffffffffffLlffffffLlffffffLlffffffLlffffffffffffffffffffffffffffffffffffffffffffffffffffffffLL",
        msgFromServer[0],
    )
    pa = np.asarray([recMSG[37],recMSG[45],recMSG[53],recMSG[61]])
    pb = np.asarray([recMSG[39],recMSG[47],recMSG[55],recMSG[63]])
    return pa,pb

def design_des_traj(pa,pb,dof,muscle):
    # Send to server using created UDP socket
    
    max_pr = 4.  # in bar
    min_pr = 1.5  # in bar
    # init pressure trajectories

    if muscle == 0:
        p = np.concatenate([np.linspace(pa[dof], pa[dof], num=500)
                        ,np.linspace(pa[dof], max_pr, num=1000)
                        ,np.linspace(min_pr, (max_pr-min_pr)/2+min_pr, num=1000)
                        ,np.linspace(max_pr, (max_pr-min_pr)*.75+min_pr, num=1000)
                        ,np.linspace((max_pr-min_pr)*.25+min_pr, min_pr, num=1000)])
        pa_des = np.zeros((4,len(p)))
        pb_des = np.zeros((4,len(p)))
        pa_des[dof,:] = p
    else:
        p = np.concatenate([np.linspace(pb[dof], pb[dof], num=500)
                        ,np.linspace(pb[dof], max_pr, num=1000)
                        ,np.linspace(min_pr, (max_pr-min_pr)/2+min_pr, num=1000)
                        ,np.linspace(max_pr, (max_pr-min_pr)*.75+min_pr, num=1000)
                        ,np.linspace((max_pr-min_pr)*.25+min_pr, min_pr, num=1000)])
        pa_des = np.zeros((4,len(p)))
        pb_des = np.zeros((4,len(p)))
        pb_des[dof,:] = p

    return pa_des,pb_des, len(p)

def send_ctrl_msg(i,ts,pa_des,pb_des,UDPClientSocket,serverAddressPort):
    myMsgCTRL = pack(
            "=HHHHLffffffffffffffffffff",
            i,  # H, index
            0,  # H, close connection
            1,  # H, 2 = position control, 1 = pressure control
            0,  # H, reserve
            int(ts),  # L, timestamp client
            1.0,  # setpoint angle J1
            1.1,  # setpoint valve J1A
            pa_des[0],  # setpoint pressure J1A
            1.3,
            pb_des[0],
            2.0,
            2.1,
            pa_des[1],  # PressureA + 1,
            2.3,
            pb_des[1],  # PressureB + 1,
            3.0,
            3.1,
            pa_des[2],  # PressureA + 2,
            3.3,
            pb_des[2],  # PressureB + 2,
            4.0,
            4.1,
            pa_des[3],  # PressureA + 3,
            4.3,
            pb_des[3],  # PressureB + 3,
        )
    UDPClientSocket.sendto(myMsgCTRL, serverAddressPort)

def send_pid_msg(P,I,UDPClientSocket,serverAddressPort,dof,muscle):
    Pas = np.asarray([793,375,375,375])
    Ias = np.asarray([12954,6000,6000,6000])
    Pbs = np.asarray([793,375,375,375])
    Ibs = np.asarray([12954,6000,6000,6000])

    if muscle == 0:
        Pas[dof] = P
        Ias[dof] = I
    else:
        Pbs[dof] = P
        Ibs[dof] = I

    myMsgPID = pack(
        "=HLLffffffffffffffffffffffffffffffffffffffffffffffff",
        0,  # H, index
        2,  # L, telegram type
        1,  # L, command
        # J1A
        Pas[0],  # f, P-Value
        Ias[0],  # f, I-Value
        0.0,  # f, D-Value
        0.01,  # f, TS
        # J1B
        Pbs[0],  # f, P-Value
        Ibs[0],  # f, I-Value
        0.0,  # f, D-Value
        0.01,  # f, TS
        # J2A
        Pas[1],  # f, P-Value
        Ias[1],  # f, I-Value
        0.0,  # f, D-Value
        0.01,  # f, TS
        # J2B
        Pbs[1],  # f, P-Value
        Ibs[1],  # f, I-Value
        0.0,  # f, D-Value
        0.01,  # f, TS
        # J3A
        Pas[2],  # f, P-Value
        Ias[2],  # f, I-Value
        0.0,  # f, D-Value
        0.01,  # f, TS
        # J3B
        Pbs[2],  # f, P-Value
        Ibs[2],  # f, I-Value
        0.0,  # f, D-Value
        0.01,  # f, TS
        # J4A
        Pas[3],  # f, P-Value
        Ias[3],  # f, I-Value
        0.0,  # f, D-Value
        0.01,  # f, TS
        # J4B
        Pbs[3],  # f, P-Value
        Ibs[3],  # f, I-Value
        0.0,  # f, D-Value
        0.01,  # f, TS
        # Pos J1
        2,  # f, P-Value
        4,  # f, I-Value
        0.1,  # f, D-Value
        0.004,  # f, TS
        # Pos J2
        3,  # f, P-Value
        6.5,  # f, I-Value
        0.1,  # f, D-Value
        0.004,  # f, TS
        # Pos J3
        5,  # f, P-Value
        10,  # f, I-Value
        0.1,  # f, D-Value
        0.004,  # f, TS
        # Pos J4
        2,  # f, P-Value
        2,  # f, I-Value
        0.01,  # f, D-Value
        0.004,  # f, TS
    )
    UDPClientSocket.sendto(myMsgPID, serverAddressPort)

def plt_results_all(pa_des,pb_des,pa,pb):
    traj_len = len(pa_des[0])
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2)   
    t = np.linspace(0,traj_len,traj_len)
    ax1.plot(t,pa_des[0],t,pa[0])
    ax2.plot(t,pb_des[0],t,pb[0])
    ax3.plot(t,pa_des[1],t,pa[1])
    ax4.plot(t,pb_des[1],t,pb[1])
    ax5.plot(t,pa_des[2],t,pa[2])
    ax6.plot(t,pb_des[2],t,pb[2])
    ax7.plot(t,pa_des[3],t,pa[3])
    ax8.plot(t,pb_des[3],t,pb[3])
    plt.show()

def plt_results(pa_des,pb_des,pa,pb,P,I,loss, dof, muscle):
    traj_len = len(pa_des[0])
    # fig, (ax1, ax2) = plt.subplots(2, 1) 
    fig = plt.figure(figsize=(6, 6))
    ax=fig.gca()
    t = np.linspace(0,traj_len,traj_len)
    if muscle == 0:
        ax.plot(t,pa_des[dof],t,pa[dof])
    else:
        ax.plot(t,pb_des[dof],t,pb[dof])
    # ax2.plot(t,pb_des[0],t,pb[0])
    plt.title(f'P: {P:3.2f} I: {I:4.2f} loss: {loss:.4f}')
    plt.ylim([1.4, 4.5])
    fn = os.path.abspath(os.curdir)+f"/figs/{-loss:06.0f}_{P:3.0f}_{I:4.0f}.jpg"
    # plt.show() 
    # plt.pause(0.1)
    plt.savefig(fn, dpi=fig.dpi)
    plt.close(fig)

    
def calc_loss(p,p_des,dof):
    N = len(p[dof,:])
    l_dev = -np.sum(np.square(p[dof,500:]-p_des[dof,500:]))
    yf = fft(p[dof,:]-p_des[dof,:])
    a_yf = np.abs(yf[0:N//2])
    l_fft = -.005*np.sum(np.square(a_yf[50:]))
    print
    plt.plot(a_yf)
    plt.grid()
    print(l_dev)
    print(l_fft)
    # plt.show(block=False) 
    # plt.pause(0.1)
    return l_dev + l_fft

def bo_fcn(P,I):
    dof = 3
    muscle = 0# 0 -> ago, 1 -> antago
    buffer_size = 1024
    serverAddressPort = ("192.168.0.110", 4700)
    UDPClientSocket = setup_connection(serverAddressPort)
    # read pressures and start at them
    pa,pb = read_pr_from_msg(UDPClientSocket,buffer_size)
    pa_des,pb_des,traj_len = design_des_traj(pa,pb,dof,muscle)
    pa = np.zeros((4,traj_len))
    pb = np.zeros((4,traj_len))
    send_pid_msg(P,I,UDPClientSocket,serverAddressPort,dof,muscle)
    for i in range(traj_len):
        ts = (time.time() % 60) * 1000
        send_ctrl_msg(i,ts,pa_des[:,i],pb_des[:,i],UDPClientSocket,serverAddressPort)
        pa[:,i],pb[:,i] = read_pr_from_msg(UDPClientSocket,buffer_size)
    if muscle == 0:
        loss = calc_loss(pa,pa_des,dof)
    else:
        loss = calc_loss(pb,pb_des,dof)
    plt_results(pa_des,pb_des,pa,pb,P,I,loss, dof, muscle)
    return loss

def main():
    pbounds = {'P': (0, 5000), 'I': (0, 25000)}
    optimizer = BayesianOptimization(
    f=bo_fcn,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=0,
    )  
    # optimizer.probe(
    #     params={"P": 700, "I": 3000},
    #     lazy=True,
    # )
    # optimizer.probe(
    #     params={"P": 657, "I": 13866},
    #     lazy=True,
    # )
    # optimizer.probe(
    #     params={"P": 599, "I": 13017},
    #     lazy=True,
    # )
    # optimizer.probe(
    #     params={"P": 573, "I": 12893},
    #     lazy=True,
    # )
    # optimizer.probe(
    #     params={"P": 570, "I": 11585},
    #     lazy=True,
    # )
    # optimizer.probe(
    #     params={"P": 463, "I": 13700},
    #     lazy=True,
    # )
    # optimizer.probe(
    #     params={"P": 592, "I": 10948},
    #     lazy=True,
    # )
    # optimizer.probe(
    #     params={"P": 474, "I": 13249},
    #     lazy=True,
    # )
    # optimizer.probe(
    #     params={"P": 683, "I": 13897},
    #     lazy=True,
    # )
    # optimizer.probe(
    #     params={"P": 606, "I": 11980},
    #     lazy=True,
    # )
    # optimizer.probe(# best found Aug 18 22 for dof 1
    #     params={"P": 793, "I": 12954},
    #     lazy=True,
    # )
    # dof 1 up to here
    optimizer.probe(
        params={"P": 561, "I": 11599},
        lazy=True,
    )
    optimizer.probe(
        params={"P": 462, "I": 7101},
        lazy=True,
    )
    optimizer.probe(
        params={"P": 472, "I": 10780},
        lazy=True,
    )
    optimizer.probe(
        params={"P": 467, "I": 7097},
        lazy=True,
    )
    optimizer.probe(
        params={"P": 432, "I": 14378},
        lazy=True,
    )
    optimizer.probe(
        params={"P": 404, "I": 8132},
        lazy=True,
    )
    optimizer.probe(# *
        params={"P": 414, "I": 5813},
        lazy=True,
    )
    optimizer.probe(
        params={"P": 425, "I": 7776},
        lazy=True,
    )
    optimizer.probe(
        params={"P": 419, "I": 7330},
        lazy=True,
    )
    optimizer.probe(
        params={"P": 415, "I": 9185},
        lazy=True,
    )
    optimizer.probe(
        params={"P": 376, "I": 5855},
        lazy=True,
    )
    optimizer.probe(
        params={"P": 375, "I": 6000},
        lazy=True,
    )
    optimizer.probe(
        params={"P": 400, "I": 5762},
        lazy=True,
    )

    optimizer.maximize(
        init_points=0,
        n_iter=50,
    )
    
    print(optimizer.max)

    optimizer = BayesianOptimization(
    f=bo_fcn,
        pbounds=pbounds,
        verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=0,
    )  
    optimizer.probe(
        params={"P": 400, "I": 5762},
        lazy=True,
    )

    optimizer.maximize(
        init_points=0,
        n_iter=0,
    )
    # input("press enter to close all figures")

    
    

    
if __name__ == "__main__":
    main()
    
