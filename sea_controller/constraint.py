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
                Kd = 1

                G = self.gravity_vector()

                taus = []
                for i in range(self.link_cnt):
                        q = self.state[(i*4)+2,0]
                        qdot = self.state[(i*4)+3,0]
                        # taus.append(max(min(Kp*(q_d[i,0]-q)+Kd*(qdot_d[i,0]-qdot),50),-50))
                        taus.append(Kp*(q_d[i,0]-q)+Kd*(qdot_d[i,0]-qdot))
                        # taus[i] += G[i]


                return np.matrix([taus]).T

        def gravity_vector(self):

                zero_list = [0]*self.link_cnt*2
                G = p.calculateInverseDynamics(self.arm,
                                               objPositions=self.get_joint_pos_list(),
                                               objVelocities=zero_list,
                                               objAccelerations=zero_list)
                G = [G[i] for i in range(len(G)) if i%2 == 1]
                return G

        def Jm(self):
                Jm = p.calculateMassMatrix(self.arm,
                                           objPositions=self.get_joint_pos_list())
                Jm = np.matrix(Jm)
                print(Jm)


        def control(self, taus):
                p.setJointMotorControlArray(self.arm,
                                            self.motorJointIDs,
                                            controlMode=p.TORQUE_CONTROL,
                                            forces=taus)


def traj(t, q0, qd):

        a0 = q0
        a1 = 0.0
        a2 = (3.0/2.0)*(qd-q0)
        a3 = (1/2)*(q0-qd)

        q_t = a0+a1*t+a2*(t**2)+a3*(t**3)
        qdot_t = a1+2*a2*t+3*a3*(t**2)
        qddot_t = 2*a2+6*a3*t

        return q_t, qdot_t, qddot_t



if __name__ == '__main__':



        time_step = 0.001

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0,0,-9.8)

        p.setRealTimeSimulation(0)
        p.setTimeStep(time_step)

        robot = SeaRobot([500,537,521])

        def step_world():
                robot.step_robot()
                p.stepSimulation()
                time.sleep(time_step)
                robot.update()

        # box = p.loadURDF("cube_small.urdf",[3,0,2],globalScaling=15)
        # p.changeDynamics(box,-1,mass=200)

        # p.loadURDF("plane.urdf", [0, 0, 0])

        tau_tracker = []
        tau_d_tracker = []

        reset = False

        def custom_loss(y, y_pred):

                loss = np.cbrt(y-y_pred)*20
                print('loss: {}'.format(np.sum(np.abs(loss))))
                return loss


        nn = Perceiver([[15,250],[250,250],[250,3]],
                       activation=[linear_unfiltered,ReLU,ReLU,linear],
                       d_activation=[d_linear_unfiltered,d_ReLU,d_ReLU,d_linear],
                       loss=custom_loss,
                       lr=0.05)



        state = np.matrix([[0]*15]).T
        goal = np.matrix([[0]*3]).T

        states = np.matrix([[0]*15]).T
        goals = np.matrix([[0]*3]).T

        plt.axis([0,10,0,1])

        plot = False

        while 1:

                t = 0
                s = [random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi)]
                g = [random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi), random.uniform(-math.pi,math.pi)]

                qd = np.matrix([g]).T

                robot.set_state(s)
                reset = False

                q0 = robot.get_link_pos()

                while t < 5:

                        if len(tau_tracker) > 0 and plot:
                                plt.clf()
                                x = np.linspace(0,len(tau_tracker),len(tau_tracker))
                                plt.plot(x,tau_tracker)
                                plt.plot(x,tau_d_tracker)
                                plt.pause(0.00001)

                        q_d, qdot_d, _ = traj(t/5, q0, qd)

                        goal = robot.get_tau_d(q_d,qdot_d)

                        state[0:12,:] = robot.get_state()
                        state[12:15,:] = goal

                        states = np.concatenate((states, state),axis=1)

                        tau, _ = nn.predict(state)

                        robot.control(tau)
                        step_world()

                        actual = robot.get_link_taus()

                        goals = np.concatenate((goals, actual),axis=1)

                        t += time_step

                nn.train(states,goals,epochs=2,batch_size=100)

        plt.show()
