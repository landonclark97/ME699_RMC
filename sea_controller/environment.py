import pybullet as p
import time
import math

import random

import numpy as np

import pybullet_data

from perceptron import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation



class SeaRobot:

        def __init__(self, k):
                self.arm = p.loadURDF("3R_sea.urdf", useFixedBase=True)

                self.k = k

                self.link_cnt = int(p.getNumJoints(self.arm)/2)
                self.motorJointIDs = [0, 2, 4]
                self.linkJointIDs = [1, 3, 5]

                # remove initial joint locks
                maxForce = 0
                mode = p.VELOCITY_CONTROL
                for i in range(p.getNumJoints(self.arm)):
                        p.setJointMotorControl2(self.arm,i,targetVelocity=0,controlMode=mode,force=maxForce)

                self.state = []
                self.link_taus = []
                self.update()

        def update(self):
                self.update_state()
                self.update_link_torques()

        def update_state(self):
                state = []

                mps = p.getJointStates(self.arm, self.motorJointIDs)
                lps = p.getJointStates(self.arm, self.linkJointIDs)

                for i in range(self.link_cnt):
                        state.append(mps[i][0])
                        state.append(mps[i][1])
                        state.append(lps[i][0])
                        state.append(lps[i][1])

                self.state = np.matrix([state]).T


        def update_link_torques(self):

                taus = []
                for i in range(self.link_cnt):
                        tau = self.k[i]*(self.state[(i*4),0]-self.state[(i*4)+2,0])
                        taus.append(tau)

                self.link_taus = np.matrix([taus]).T


        def get_state(self):
                return self.state

        def get_link_taus(self):
                return self.link_taus

        def get_link_state(self):
                pos = []
                vel = []
                for i in range(self.link_cnt):
                        pos.append(self.state[(i*4)+2,0])
                        vel.append(self.state[(i*4)+3,0])
                return np.matrix([pos+vel]).T

        def reset_state(self):
                for i in range(self.link_cnt*2):
                        p.resetJointState(self.arm,i,targetValue=0,targetVelocity=0)
                self.update()

        def set_state(self, val):
                for i in range(self.link_cnt*2):
                        p.resetJointState(self.arm,i,targetValue=val[i//2],targetVelocity=0)
                self.update()

        def step_robot(self):
                self.update()

                taus = []
                for i in range(self.link_cnt):
                        taus.append(self.link_taus[i,0])

                p.setJointMotorControlArray(self.arm,
                                            self.linkJointIDs,
                                            controlMode=p.TORQUE_CONTROL,
                                            forces=taus)

                taus = [-tau for tau in taus]

                p.setJointMotorControlArray(self.arm,
                                            self.motorJointIDs,
                                            controlMode=p.TORQUE_CONTROL,
                                            forces=taus)

        def get_joint_pos_list(self):
                pos = []
                for i in range(self.link_cnt):
                        pos.append(self.state[(i*4),0])
                        pos.append(self.state[(i*4)+2,0])
                return pos

        def get_link_pos(self):
                pos = self.get_joint_pos_list()
                pos = [pos[i] for i in range(len(pos)) if i%2 != 0]
                return np.matrix([pos]).T

        def get_tau_d(self, q_d, qdot_d):
                Kp = 500
                Kd = 0.5

                G = self.gravity_vector()

                taus = []
                for i in range(self.link_cnt):
                        q = self.state[(i*4)+2,0]
                        qdot = self.state[(i*4)+3,0]
                        taus.append(max(min(Kp*(q_d[i,0]-q)+Kd*(qdot_d[i,0]-qdot),100),-100))
                        # taus.append(Kp*(q_d[i,0]-q)+Kd*(qdot_d[i,0]-qdot))
                        # taus[i] += G[i]


                return np.matrix([taus]).T

        def control(self, taus):
                p.setJointMotorControlArray(self.arm,
                                            self.motorJointIDs,
                                            controlMode=p.TORQUE_CONTROL,
                                            forces=taus)




## ENVIRONMENT INIT: ########################
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0,0,-9.8)

p.setRealTimeSimulation(0)
time_step = 0.001
p.setTimeStep(time_step)

robot = SeaRobot([500,537,521])
t = 0
traj_time = 2



# s = [random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi)]
# g = [random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi)]
s = [0, 0, 0]
g = [0, -math.pi/2, math.pi/4]

q0 = robot.get_link_pos()
qd = np.matrix([g]).T

robot.set_state(s)
#############################################



def traj(t, q0, qd):

        a0 = q0
        a1 = 0.0
        a2 = (3.0/2.0)*(qd-q0)
        a3 = (1/2)*(q0-qd)

        q_t = a0+a1*t+a2*(t**2)+a3*(t**3)
        qdot_t = a1+2*a2*t+3*a3*(t**2)
        qddot_t = 2*a2+6*a3*t

        return q_t, qdot_t, qddot_t



def refresh_env():
        global t
        global s
        global g
        global q0
        global qd

        t = 0

        # s = [random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi)]
        # g = [random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi)]
        s = [0, 0, 0]
        g = [0, -math.pi/2, math.pi/4]

        q0 = robot.get_link_pos()
        qd = np.matrix([g]).T

        robot.set_state(s)


def reward(y, y_hat):
        summ = np.sum(np.abs(y[0:3]-y_hat[0:3]))
        if summ == 0.0:
                summ = 0.001
        return 10*(1.0/summ)-2.5


def env_state():
        state = np.matrix([[0]*12],dtype='float64').T
        q_d, qdot_d, _ = traj(t/traj_time, q0, qd)
        goal = np.matrix([q_d.T.tolist()[0]+qdot_d.T.tolist()[0]]).T

        state[0:12,0] = robot.get_state()
        # state[12:18,0] = goal
        # state[12:15,0] = qd

        state = np.squeeze(np.asarray((state)))
        return state


def step_env(x):
        global t

        q_d, qdot_d, _ = traj(t/traj_time, q0, qd)

        # goal = np.matrix([q_d.T.tolist()[0]+qdot_d.T.tolist()[0]]).T
        goal = qd

        done = False

        robot.control(x)
        robot.step_robot()
        p.stepSimulation()
        robot.update()

        t += time_step

        next_state = env_state()
        actual = robot.get_link_state()


        if t >= traj_time:
                done = True
                refresh_env()

        r = reward(actual, goal)

        return next_state, r, done
