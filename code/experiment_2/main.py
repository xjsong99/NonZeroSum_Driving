import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import math
import numpy as np
import pandas as pd
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(3)
np.random.seed(3)
random.seed(3)

class Dynamic:
    def __init__(self, args):
        # state = [X1, Y1, e_d, e_phi, r, v_y, X2, Y2]
        self.args = args
        self.deltaT = args.deltaT
        self.kf, self.kr = -88000., -94000.
        self.lf, self.lr = 1.14, 1.4
        self.m = 1500.
        self.Iz = 2420.

    def transform(self, P, P0, phi):
        '''
        <input>
        P       :tensor[2,1] #[X, Y]
        P0      :tensor[2,1] #[X0, Y0]
        phi     :float
        <output>
        P_veh   :tensor[2,1] #[x_veh, y_veh]
        '''
        P -= P0
        R = torch.zeros([2, 2])
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = torch.cos(-phi), -torch.sin(-phi), torch.sin(-phi), torch.cos(-phi)
        return torch.mm(R,P)

    def step_virtual(self, state, u1, u2, agent_1, agent_2):
        '''
        <input>
        state     :tensor[agent_size, 9]
        delta_f   :tensor[agent_size, 1]
        tra_X     :list[]
        tra_Y     :list[]
        <return>
        state     :tensor[agent_size, 9]  #[X1, Y1, phi1, e_d, e_phi, r, v_y, X2, Y2]
        '''
        # state = [X1, Y1, d1, theta1, u1, v1, w1, X2, Y2, d2, theta2, u2, v2, w2]
        # index = [0,  1,  2,  3,      4,  5,  6,  7,  8,  9,  10,     11, 12, 13]

        # u     = [a, δ]
        # index = [0, 1]

        # 更新状态state
        new_state = []
        for i in range(14):
            new_state.append([])

        new_state[0] = state[:, 0] + self.args.deltaT * (state[:, 4] * torch.cos(state[:, 3]) - state[:, 5] * torch.sin(state[:, 3]))
        new_state[1] = state[:, 1] + self.args.deltaT * (state[:, 4] * torch.sin(state[:, 3]) + state[:, 5] * torch.cos(state[:, 3]))
        new_state[2] = state[:, 2] + self.args.deltaT * (state[:, 4] * torch.sin(state[:, 3]) + state[:, 5] * torch.cos(state[:, 3]))
        new_state[3] = state[:, 3] + self.args.deltaT * state[:, 6]
        new_state[4] = state[:, 4] + self.args.deltaT * u1[:, 0]
        new_state[5] = (self.m * state[:, 4] * state[:, 5] + self.args.deltaT * (self.lf * self.kf - self.lr * self.kr) * state[:, 6] - self.args.deltaT * self.kf * u1[:, 1] * state[:, 4] - self.args.deltaT * self.m * torch.pow(state[:, 4], 2) * state[:, 6]) \
                       / (self.m * state[:, 4] - self.deltaT * (self.kf + self.kr))
        new_state[6] = (self.Iz * state[:, 4] * state[:, 6] + self.deltaT * (self.lf * self.kf - self.lr * self.kr) * state[:, 5] - self.deltaT * self.lf * self.kf * u1[:, 1] * state[:, 4])\
                       / (self.Iz * state[:, 4] - self.args.deltaT * (self.lf * self.lf * self.kf + self.lr * self.lr * self.kr))
        
        new_state[7] = state[:, 7] + self.args.deltaT * (state[:, 11] * torch.cos(state[:, 10] + math.pi / 2) - state[:, 12] * torch.sin(state[:, 10] + math.pi / 2))
        new_state[8] = state[:, 8] + self.args.deltaT * (state[:, 11] * torch.sin(state[:, 10] + math.pi / 2) + state[:, 12] * torch.cos(state[:, 10] + math.pi / 2))
        new_state[9] = state[:, 9] - self.args.deltaT * (state[:, 11] * torch.cos(state[:, 10] + math.pi / 2) - state[:, 12] * torch.sin(state[:, 10] + math.pi / 2))
        new_state[10] = state[:, 10] + self.args.deltaT * state[:, 13]
        new_state[11] = state[:, 11] + self.args.deltaT * u2[:, 0]
        new_state[12] = (self.m * state[:, 11] * state[:, 12] + self.args.deltaT * (self.lf * self.kf - self.lr * self.kr) * state[:, 13] - self.args.deltaT * self.kf * u2[:, 1] * state[:, 11] - self.args.deltaT * self.m * torch.pow(state[:, 11], 2) * state[:, 13]) \
                        / (self.m * state[:, 4] - self.deltaT * (self.kf + self.kr))
        new_state[13] = (self.Iz * state[:, 11] * state[:, 13] + self.deltaT * (self.lf * self.kf - self.lr * self.kr) * state[:, 12] - self.deltaT * self.lf * self.kf * u2[:, 1] * state[:, 11])\
                        / (self.Iz * state[:, 11] - self.args.deltaT * (self.lf * self.lf * self.kf + self.lr * self.lr * self.kr))

        for i in range(14):
            new_state[i] = new_state[i].unsqueeze(1)

        return torch.cat(new_state, 1)
    
    def check_done(self, state, agent_1, agent_2):
        for i in range(self.args.agent_size):
            if torch.abs(state[i, 2]) > self.args.d_lim or \
               torch.abs(state[i, 9]) > self.args.d_lim or \
               torch.abs(state[i, 3]) > self.args.phi_lim or \
               torch.abs(state[i, 10]) > self.args.phi_lim or \
               state[i, 4] < 0 or state[i, 11] < 0 or \
               state[i, 4] > self.args.u_lim or state[i, 11] > self.args.u_lim or \
               state[i, 0] > agent_1.tra_point[-1][0, 0] or \
               state[i, 8] > agent_2.tra_point[-1][1, 0]:
                print("【Hint】Check Done: [",end="")
                for j in range(14):
                    print("%6.2f" % float(state[i, j]), end=",")
                print("]")
                rand_num = random.uniform(0, agent_1.tra_point[-1][0, 0] - agent_1.tra_point[0][0, 0])
                state[i, 0] = agent_1.tra_point[0][0, 0] + rand_num # X1
                state[i, 1] = agent_1.tra_point[0][1, 0] + random.uniform(-2, 2)  # Y1
                state[i, 2] = state[i, 1] - agent_1.tra_point[0][1, 0]  # d1
                state[i, 3] = 0  # theta1
                state[i, 4] = agent_1.v_x + random.uniform(-1, 1)  # u1
                state[i, 5] = random.uniform(-0.05, 0.05)  # v1
                state[i, 6] = random.uniform(-0.05, 0.05)  # w1

                state[i, 7] = agent_2.tra_point[0][0, 0] + random.uniform(-2, 2) # X2
                state[i, 8] = agent_2.tra_point[0][1, 0] + rand_num  # Y2
                state[i, 9] = -(state[i, 7] - agent_2.tra_point[0][0, 0])  # d2
                state[i, 10] = 0  # theta2
                state[i, 11] = agent_2.v_x + random.uniform(-1, 1) # u2
                state[i, 12] = random.uniform(-0.05, 0.05)  # v2
                state[i, 13] = random.uniform(-0.05, 0.05)  # w2
        return state

class Agent:
    def __init__(self, args, vehicle_type):
        self.args = args

        # trajectory_points:[tensor[X0,Y0],tensor[X1,Y1],...]    #要跟踪的大地坐标点
        self.tra_point = []

        self.v_x = 0
        self.kappa = 0
        self.vehicle_type = vehicle_type # 1:西->东  2:南->北

        # 敌方参数
        self.tra_point_enemy = []
        self.v_x_enemy = 0
        self.kappa_enemy = 0

        '''
        # 这套参数能收敛且跟踪轨迹，但不能避撞，跟踪参考速度的性能也有待提高
        self.a1, self.a2, self.a3 = 1.0 / (self.args.d_lim ** 2), 0.08 / (self.args.phi_lim ** 2), 0.01 / (self.args.u_lim ** 2)
        self.a4, self.a5 = 0.002 / (self.args.max_angle ** 2), 0.001 / (self.args.max_acc ** 2)
        self.a6 = 1.25 / (self.args.dis_range ** 2)
        '''

        '''
        # 这套参数能收敛且跟踪轨迹，能避撞但措施都是打方向盘而不是减速，跟踪参考速度的性能也有待提高
        self.a1, self.a2, self.a3 = 1.0 / (self.args.d_lim ** 2), 0.08 / (self.args.phi_lim ** 2), 0.1 / (self.args.u_lim ** 2)
        self.a4, self.a5 = 0.002 / (self.args.max_angle ** 2), 0.001 / (self.args.max_acc ** 2)
        self.a6 = 5.0 / (self.args.dis_range ** 2)
        '''

        '''
        # 这套参数能收敛且跟踪轨迹，能避撞但措施是打方向盘和稍微减速，跟踪参考速度的性能也有待提高
        self.a1, self.a2, self.a3 = 1.0 / (self.args.d_lim ** 2), 0.08 / (self.args.phi_lim ** 2), 0.5 / (self.args.u_lim ** 2)
        self.a4, self.a5 = 0.002 / (self.args.max_angle ** 2), 0.0001 / (self.args.max_acc ** 2)
        self.a6 = 5.0 / (self.args.dis_range ** 2)
        '''

        '''
        # 这套参数能收敛且跟踪轨迹，避撞措施是打方向盘和稍微减速，跟踪参考速度的性能较上面的参数更好，但跟踪参考速度时调节太慢
        self.a1, self.a2, self.a3 = 1.5 / (self.args.d_lim ** 2), 0.1 / (self.args.phi_lim ** 2), 5.0 / (self.args.u_lim ** 2)
        self.a4, self.a5 = 0.002 / (self.args.max_angle ** 2), 0.00001 / (self.args.max_acc ** 2)
        self.a6 = 5.0 / (self.args.dis_range ** 2)
        '''

        # 比较好的参数
        self.a1, self.a2, self.a3 = 1.5 / (self.args.d_lim ** 2), 0.1 / (self.args.phi_lim ** 2), 5.0 / ((self.args.u_lim / 2)** 2)
        self.a4, self.a5 = 0.002 / (self.args.max_angle ** 2), 0.0001 / (self.args.max_acc ** 2)
        self.a6 = 5.0 / (self.args.dis_range ** 2)

        self.actor_net = nn.Sequential(
            nn.Linear(14, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, 2),
            nn.Tanh()                #别忘乘args.max_angle(default=0.35)
        ).cuda()
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), self.args.lr_actor)
        # self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.99)
        for m in self.actor_net.modules():
            if isinstance(m, nn.Linear):
                # torch.nn.init.uniform_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)

        self.critic_net = nn.Sequential(
            nn.Linear(14, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, 1),
            nn.ReLU(),
        ).cuda()
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), self.args.lr_critic)
        # self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.99)
        for m in self.critic_net.modules():
            if isinstance(m, nn.Linear):
                # torch.nn.init.uniform_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)

        if args.mode=="train" and args.load_old_network == "True":
            self.load_network(-1)

    def forward_actor(self, x):
        if self.vehicle_type == 1:
            v_x_1 = self.v_x
            v_x_2 = self.v_x_enemy
        else:
            v_x_1 = self.v_x_enemy
            v_x_2 = self.v_x
        norm_matrix = torch.tensor([1.0 / 100, 1.0 / self.args.d_lim, 1.0 / self.args.d_lim, 1 / self.args.phi_lim, 1 / v_x_1, 1, 1, \
                                    1.0 / self.args.d_lim, 1.0 / 100, 1.0 / self.args.d_lim, 1 / self.args.phi_lim, 1 / v_x_2, 1, 1], dtype=torch.float32).cuda()
        x = torch.mul(x, norm_matrix)
        gain_matrix = torch.zeros([1, 2]).cuda()
        gain_matrix[0, 0], gain_matrix[0, 1] = self.args.max_acc, self.args.max_angle
        x = gain_matrix * self.actor_net(x)
        return x

    def forward_critic(self, x):
        if self.vehicle_type == 1:
            v_x_1 = self.v_x
            v_x_2 = self.v_x_enemy
        else:
            v_x_1 = self.v_x_enemy
            v_x_2 = self.v_x
        norm_matrix = torch.tensor([1.0 / 100, 1.0 / self.args.d_lim, 1.0 / self.args.d_lim, 1 / self.args.phi_lim, 1 / v_x_1, 1, 1, \
                                    1.0 / self.args.d_lim, 1.0 / 100, 1.0 / self.args.d_lim, 1 / self.args.phi_lim, 1 / v_x_2, 1, 1], dtype=torch.float32).cuda()
        x = torch.mul(x, norm_matrix)
        x = self.critic_net(x)
        return x

    def utility(self, state, u):
        '''
        <input>
        state     :[agent_size, 16]
        u         :[agent_size, 2]
        <return>
        l         :[agent_size, 1]
        '''
        # state = [X1, Y1, d1, theta1, u1, v1, w1, X2, Y2, d2, theta2, u2, v2, w2]
        # index = [0,  1,  2,  3,      4,  5,  6,  7,  8,  9,  10,     11, 12, 13]

        # u     = [a, δ]
        # index = [0, 1]

        if self.vehicle_type == 1:
            D_value = self.args.dis_range**2 - (torch.pow(state[:, 0] - state[:, 7], 2) + torch.pow(state[:, 1] - state[:, 8], 2)).unsqueeze(1)
            D_value = torch.where(D_value < 0, torch.zeros([1,1]).cuda(), D_value)  # D_value[D_value < 0] = 0
            l = (self.a1 * torch.pow(state[:, 2], 2)).unsqueeze(1) + \
                (self.a2 * torch.pow(state[:, 3], 2)).unsqueeze(1) + \
                (self.a3 * torch.pow(state[:, 4] - self.v_x, 2)).unsqueeze(1) + \
                self.a4 * torch.pow(u[:, 0], 2) + \
                self.a5 * torch.pow(u[:, 1], 2) + \
                self.a6 * D_value
        else:
            D_value = self.args.dis_range**2 - (torch.pow(state[:, 0] - state[:, 7], 2) + torch.pow(state[:, 1] - state[:, 8], 2)).unsqueeze(1)
            D_value = torch.where(D_value < 0, torch.zeros([1,1]).cuda(), D_value)  # D_value[D_value < 0] = 0
            l = (self.a1 * torch.pow(state[:, 9], 2)).unsqueeze(1) + \
                (self.a2 * torch.pow(state[:, 10], 2)).unsqueeze(1) + \
                (self.a3 * torch.pow(state[:, 11] - self.v_x, 2)).unsqueeze(1) + \
                self.a4 * torch.pow(u[:, 0], 2) + \
                self.a5 * torch.pow(u[:, 1], 2) + \
                self.a6 * D_value
        return l

    def save_network(self, iter_index):
        path = "./data/agent_"+str(self.vehicle_type)+"/actor_iter_" + str(iter_index)
        torch.save(self.actor_net.state_dict(), path)
        path = "./data/agent_"+str(self.vehicle_type)+"/critic_iter_" + str(iter_index)
        torch.save(self.critic_net.state_dict(), path)
    
    def load_network(self, check_point_index):
        iter_index = 0
        if check_point_index == -1:
            while os.path.isfile("./data/agent_"+str(self.vehicle_type)+"/actor_iter_" + str(iter_index + 100)):
                iter_index += 100
        else:
            iter_index = check_point_index
        print("agent", self.vehicle_type, "load data from iter_index = ", iter_index)
        path = "./data/agent_"+str(self.vehicle_type)+"/actor_iter_" + str(iter_index)
        self.actor_net.load_state_dict(torch.load(path))
        path = "./data/agent_"+str(self.vehicle_type)+"/critic_iter_" + str(iter_index)
        self.critic_net.load_state_dict(torch.load(path))

    def load_trajectory(self, path):
        file_trajectory = open(path, "r")
        data = file_trajectory.readlines()
        for i in range(len(data) - 1):
            point = data[i].split()
            point_tensor = torch.zeros([2, 1])
            point_tensor[0, 0], point_tensor[1, 0] = float(point[0]), float(point[1])
            self.tra_point.append(point_tensor)
        self.v_x = float(data[-1].split()[0])
        self.kappa = float(data[-1].split()[1])
    
    def ref_traj(self, t):
        return self.tra_point[0] * ((1 - t)** 3) + 3 * self.tra_point[1] * t * ((1 - t)** 2) + 3 * self.tra_point[2] * (t ** 2) * (1 - t) + self.tra_point[3] * (t ** 3)
    
    def ref_phi(self, t):
        grad = -3 * self.tra_point[0] * ((1 - t)** 2) + 3 * self.tra_point[1] * ((1 - t)** 2) \
               -6 * self.tra_point[1] * t * (1 - t) + 6 * self.tra_point[2] * t * (1 - t) \
               -3 * self.tra_point[2] * (t ** 2) + 3 * self.tra_point[3] * (t ** 2)
        return torch.atan(grad[1, 0] / grad[0, 0])

class Buffer:
    def __init__(self, args):
        self.args = args
        self.data_pool = []
        self.p_index = -1
    
    def add(self, state):
        for i in range(self.args.agent_size):
            self.p_index = (self.p_index + 1) % self.args.buffer_size
            if len(self.data_pool) < self.args.buffer_size:
                self.data_pool.append(state[i,:].clone().detach())
            else:
                self.data_pool[self.p_index] = state[i,:].clone().detach()
    
    def sample(self):
        if len(self.data_pool) < self.args.batch_size:
            return []
        index = [random.randint(0, len(self.data_pool) - 1) for _ in range(self.args.batch_size)]
        return torch.cat([self.data_pool[i].unsqueeze(0) for i in index], 0)

class Train:
    def __init__(self, args, agent_1, agent_2, env):
        self.args = args
        self.value_loss_1_record = []
        self.policy_loss_1_record = []
        self.value_loss_2_record = []
        self.policy_loss_2_record = []

        self.buffer = Buffer(args)
        self.mini_batch = []

        # state = [X1, Y1, d1, theta1, u1, v1, w1, X2, Y2, d2, theta2, u2, v2, w2]
        # index = [0,  1,  2,  3,      4,  5,  6,  7,  8,  9,  10,     11, 12, 13]

        self.state = torch.zeros([self.args.agent_size, 14])
        self.state[:, 2] = torch.normal(0.0, torch.full([args.agent_size,], 2)) # ed1
        self.state[:, 1] = self.state[:, 2] + agent_1.tra_point[0][1, 0]  # Y1
        self.state[:, 0] = torch.linspace(agent_1.tra_point[0][0,0],agent_1.tra_point[-1][0,0],args.agent_size) # agent.tra_point[0][0, 0]  # X1
        self.state[:, 4] = agent_1.v_x + torch.normal(0.0, torch.full([args.agent_size,], 1)) # u1
        self.state[:, 5] = torch.normal(0.0, torch.full([args.agent_size,], 0.05)) # vy
        self.state[:, 6] = torch.normal(0.0, torch.full([args.agent_size,], 0.05))  # r

        self.state[:, 9] = torch.normal(0.0, torch.full([args.agent_size,], 2)) # ed2
        self.state[:, 7] = -self.state[:, 9] + agent_2.tra_point[0][0, 0]  # X2
        self.state[:, 8] = torch.linspace(agent_2.tra_point[0][1,0],agent_2.tra_point[-1][1,0],args.agent_size) # agent.tra_point[0][0, 0]  # Y2
        self.state[:, 11] = agent_2.v_x + torch.normal(0.0, torch.full([args.agent_size,], 1)) # u2
        self.state[:, 12] = torch.normal(0.0, torch.full([args.agent_size,], 0.05)) # vy
        self.state[:, 13] = torch.normal(0.0, torch.full([args.agent_size,], 0.05))  # r

        self.state = self.state.cuda()
        self.state.detach_()
        
        self.mini_batch_forward = []
        self.u_1_forward = []
        self.u_2_forward = []
        self.l_1_forward = []
        self.l_2_forward = []
        for i in range(args.step_number):
            self.mini_batch_forward.append([])
            self.u_1_forward.append([])
            self.u_2_forward.append([])
            self.l_1_forward.append([])
            self.l_2_forward.append([])
        self.mini_batch_forward.append([])

    def save_args(self):
        parameter_file = open("./data/train-parameter","w")
        parameter_file.write(self.args.__str__())
        parameter_file.close()
    
    def state_update(self, env, agent_1, agent_2):
        # self.draw_state(self.state_1, self.state_1_forward, agent)

        # 计算控制量
        self.agent_u1 = agent_1.forward_actor(self.state)
        self.agent_u2 = agent_2.forward_actor(self.state)

        # 计算下一状态state_next
        state_next = env.step_virtual(self.state, self.agent_u1, self.agent_u2, agent_1, agent_2)

        self.state = state_next.clone().detach()
        self.state = env.check_done(self.state, agent_1, agent_2)

        self.buffer.add(self.state)
        
    def sample_from_buffer(self):
        self.mini_batch = self.buffer.sample()

    def value_update(self, env, agent_1, agent_2):
        if len(self.mini_batch) == 0:
            return 0, 0

        # 计算J(x(0))
        J_current_1 = agent_1.forward_critic(self.mini_batch)
        J_current_2 = agent_2.forward_critic(self.mini_batch)

        # 计算l_sum=l(x(0))+l(x(1))+...+l(x(0+step_number-1))
        self.mini_batch_forward[0] = self.mini_batch.detach()
        gamma_now = self.args.gamma
        for i in range(self.args.step_number):
            self.u_1_forward[i] = agent_1.forward_actor(self.mini_batch_forward[i].detach())
            self.u_2_forward[i] = agent_2.forward_actor(self.mini_batch_forward[i].detach())
            self.l_1_forward[i] = gamma_now * agent_1.utility(self.mini_batch_forward[i], self.u_1_forward[i])
            self.l_2_forward[i] = gamma_now * agent_2.utility(self.mini_batch_forward[i], self.u_2_forward[i])
            self.mini_batch_forward[i + 1] = env.step_virtual(self.mini_batch_forward[i], self.u_1_forward[i], self.u_2_forward[i], agent_1, agent_2)
            gamma_now = gamma_now * self.args.gamma

        self.l_sum_1 = torch.sum(torch.cat(self.l_1_forward, 1), 1)
        self.l_sum_2 = torch.sum(torch.cat(self.l_2_forward, 1), 1)

        # 计算J(x(0+step_number))
        J_step_number_1 = agent_1.forward_critic(self.mini_batch_forward[-1])
        J_step_number_2 = agent_2.forward_critic(self.mini_batch_forward[-1])

        target_value_1 = self.l_sum_1.detach() + gamma_now * J_step_number_1.detach()  # 带V
        target_value_2 = self.l_sum_2.detach() + gamma_now * J_step_number_2.detach()  # 带V
        # target_value = self.l_sum.detach() # 不带V

        current_value_1 = J_current_1
        current_value_2 = J_current_2

        
        # 无正则化
        value_loss_1 = 1 / 2 * torch.mean(torch.pow(target_value_1 - current_value_1, 2))
        value_loss_2 = 1 / 2 * torch.mean(torch.pow(target_value_2 - current_value_2, 2))
        
        '''
        # 正则化
        equilibrium_state = torch.tensor([agent_1.tra_point[-1][0, 0], 0.0, 0.0, 0.0, 0.0, 0.0, \
                                          0.0, agent_2.tra_point[-1][1, 0], 0.0, 0.0, 0.0, 0.0]).cuda()
        value_equilibrium_1 = agent_1.forward_critic(equilibrium_state)
        value_equilibrium_2 = agent_2.forward_critic(equilibrium_state)
        value_loss_1 = 1 / 2 * torch.mean(torch.pow(target_value_1 - current_value_1, 2)) + 100 * torch.pow(value_equilibrium_1, 2)
        value_loss_2 = 1 / 2 * torch.mean(torch.pow(target_value_2 - current_value_2, 2)) + 100 * torch.pow(value_equilibrium_2, 2)
        '''

        agent_1.critic_optimizer.zero_grad()
        agent_2.critic_optimizer.zero_grad()

        value_loss_1.backward()
        value_loss_2.backward()
        
        # torch.nn.utils.clip_grad_norm_(agent.critic_net.parameters(), 0.1)
        agent_1.critic_optimizer.step()
        agent_2.critic_optimizer.step()
        # agent.critic_scheduler.step()
        
        # 记录value_loss的变化过程
        self.value_loss_1_record.append(float(value_loss_1))
        self.value_loss_2_record.append(float(value_loss_2))
        return float(value_loss_1), float(value_loss_2)
        
    def policy_update(self, agent_1, agent_2):
        if len(self.mini_batch) == 0:
            return 0, 0

        # 带V
        J_step_number_1 = agent_1.forward_critic(self.mini_batch_forward[-1])
        J_step_number_2 = agent_2.forward_critic(self.mini_batch_forward[-1])

        policy_loss_1 = torch.mean(self.l_sum_1 + J_step_number_1)
        policy_loss_2 = torch.mean(self.l_sum_2 + J_step_number_2)

        agent_1.actor_optimizer.zero_grad()
        agent_2.actor_optimizer.zero_grad()
        policy_loss_1.backward(retain_graph=True)
        agent_1.actor_optimizer.step()

        agent_1.actor_optimizer.zero_grad()
        agent_2.actor_optimizer.zero_grad()
        policy_loss_2.backward()
        agent_2.actor_optimizer.step()

        # 不带V
        # policy_loss = torch.mean(self.l_sum)

        # torch.nn.utils.clip_grad_norm_(agent.actor_net.parameters(), 0.1)
        # agent.actor_scheduler.step()
        
        self.policy_loss_1_record.append(float(policy_loss_1))
        self.policy_loss_2_record.append(float(policy_loss_2))
        return float(policy_loss_1), float(policy_loss_2)

    def draw_state(self, state, state_forward, agent):
        tmp_x = [float(i[0, 0]) for i in agent.tra_point]
        tmp_y = [float(i[1, 0]) for i in agent.tra_point]
        f = plt.figure()
        f.canvas.manager.window.move(900, 300)
        plt.plot(tmp_x, tmp_y, linestyle='-', linewidth=2, color="green")
        
        if len(state_forward[0]) > 0:
            tmp_x = [float(i[0, 0]) for i in state_forward]
            tmp_y = [float(i[0, 1]) for i in state_forward]
        plt.plot(tmp_x, tmp_y, linestyle='-', linewidth=2, color="orange")
        
        plt.scatter(float(state[0, 0]), float(state[0, 1]), c="orange")
        plt.plot([float(state[0, 0]), float(state[0, 0]) + float(torch.cos(state[0, 2]))], \
            [float(state[0, 1]), float(state[0, 1]) + float(torch.sin(state[0, 2]))], linestyle='-', linewidth=1, color="black")

        plt.pause(0.1)
        plt.close()


    def render(self):
        plt.figure()
        plt.plot(range(len(self.value_loss_1_record)), self.value_loss_1_record, label="value_loss_1")
        plt.plot(range(len(self.value_loss_2_record)), self.value_loss_2_record, label="value_loss_2")
        plt.plot(range(len(self.policy_loss_1_record)), self.policy_loss_1_record, label="policy_loss_1")
        plt.plot(range(len(self.policy_loss_2_record)), self.policy_loss_2_record, label="policy_loss_2")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("./data/Loss.png")
        plt.show()

class Test:
    def __init__(self, args):
        self.args = args
        args.agent_size = 1

        parameter_file = open("./data/test-parameter","w")
        parameter_file.write(self.args.__str__())
        parameter_file.close()
        
        self.env = Dynamic(args)

        self.agent_1 = Agent(args, 1)
        self.agent_1.load_network(args.check_point)
        self.agent_1.load_trajectory("./data/ref_traj_1")

        self.agent_2 = Agent(args, 2)
        self.agent_2.load_network(args.check_point)
        self.agent_2.load_trajectory("./data/ref_traj_2")

        self.agent_1.tra_point_enemy = self.agent_2.tra_point
        self.agent_2.tra_point_enemy = self.agent_1.tra_point
        self.agent_1.v_x_enemy = self.agent_2.v_x
        self.agent_2.v_x_enemy = self.agent_1.v_x
        self.agent_1.kappa_enemy = self.agent_2.kappa
        self.agent_2.kappa_enemy = self.agent_1.kappa

        # state = [X1, Y1, d1, theta1, u1, v1, w1, X2, Y2, d2, theta2, u2, v2, w2]
        # index = [0,  1,  2,  3,      4,  5,  6,  7,  8,  9,  10,     11, 12, 13]

        self.state = torch.zeros([self.args.agent_size, 14])
        self.state[:, 2] = 5
        self.state[:, 1] = self.state[:, 2] + self.agent_1.tra_point[0][1, 0]
        self.state[:, 0] = self.agent_1.tra_point[0][0, 0]
        self.state[:, 4] = 5.5
        self.state[:, 5] = torch.normal(0.0, torch.full([args.agent_size,], 0.05))
        self.state[:, 6] = torch.normal(0.0, torch.full([args.agent_size,], 0.05))

        self.state[:, 9] = -10
        self.state[:, 7] = -self.state[:, 9] + self.agent_2.tra_point[0][0, 0] 
        self.state[:, 8] = self.agent_2.tra_point[0][1, 0]
        self.state[:, 11] = 4.5
        self.state[:, 12] = torch.normal(0.0, torch.full([args.agent_size,], 0.05))
        self.state[:, 13] = torch.normal(0.0, torch.full([args.agent_size,], 0.05))
            
        self.state = self.state.cuda()

    def run(self):
        X1_history = [float(self.state[0, 0]),]
        Y1_history = [float(self.state[0, 1]),]
        X2_history = [float(self.state[0, 7]),]
        Y2_history = [float(self.state[0, 8]),]
        utility_1 = []
        utility_2 = []
        u_speed_1 = []
        u_speed_2 = []
        dis = []
        time = []

        iter_index = 0
        while X1_history[-1] < float(self.agent_1.tra_point[-1][0, 0]) and \
              Y2_history[-1] < float(self.agent_2.tra_point[-1][1, 0]):
            if iter_index > 1000:
                break
            if iter_index % 50 == 0:
                print("iter = ", iter_index)
            
            time.append(iter_index * self.args.deltaT)
            
            iter_index += 1
            
            u1 = self.agent_1.forward_actor(self.state)
            u2 = self.agent_2.forward_actor(self.state)

            utility_1.append(float(self.agent_1.utility(self.state, u1)))
            utility_2.append(float(self.agent_2.utility(self.state, u2)))

            u_speed_1.append(float(self.state[0, 4]))
            u_speed_2.append(float(self.state[0, 11]))
            '''
            print("    s=[", end="")
            for i in range(14):
                print("%6.2f"%float(self.state[0, i]), end=",")
            print("]\tu1=[%6.2f, %6.2f]" % (float(u1[0, 0]), float(u1[0, 1])), end="")
            print("\tu2=[%6.2f, %6.2f]" % (float(u2[0, 0]), float(u2[0, 1])))
            '''
            self.state = self.env.step_virtual(self.state, u1, u2, self.agent_1, self.agent_2)

            X1_history.append(float(self.state[0, 0]))
            Y1_history.append(float(self.state[0, 1]))
            X2_history.append(float(self.state[0, 7]))
            Y2_history.append(float(self.state[0, 8]))
        X1_history.pop()
        X2_history.pop()
        Y1_history.pop()
        Y2_history.pop()

        # 画轨迹
        plt.figure()
        sns.set(style="dark")
        plt.xlim([self.agent_1.tra_point[0][0, 0], self.agent_1.tra_point[-1][0, 0]])
        plt.ylim([self.agent_2.tra_point[0][1, 0], self.agent_2.tra_point[-1][1, 0]])
        # plot reference
        tmp_x = [float(i[0, 0]) for i in self.agent_1.tra_point]
        tmp_y = [float(i[1, 0]) for i in self.agent_1.tra_point]
        plt.plot(tmp_x, tmp_y, linestyle='dashed', linewidth=1, color="green", label='reference')
        tmp_x = [float(i[0, 0]) for i in self.agent_2.tra_point]
        tmp_y = [float(i[1, 0]) for i in self.agent_2.tra_point]
        plt.plot(tmp_x, tmp_y, linestyle='dashed', linewidth=1, color="green")

        plt.plot(X1_history, Y1_history, label='vehicle 1')
        plt.plot(X2_history, Y2_history, label='vehicle 2')
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        # plt.gca().set_aspect("equal")
        sns.despine(top=True, right=True, left=True, bottom=True)
        plt.tight_layout(pad=0.5)
        plt.savefig("./data/test_traj.png")

        # 画速度
        plt.figure()
        sns.set(style="darkgrid", font_scale=1)
        plt.ylim([4,6])
        df = pd.DataFrame({'time': time, 'speed':u_speed_1})
        sns.lineplot(x="time", y="speed", data=df, legend='brief', label="vehicle 1")
        df = pd.DataFrame({'time': time, 'speed':u_speed_2})
        sns.lineplot(x="time", y="speed", data=df, legend='brief', label="vehicle 2")
        plt.xlabel("time [s]")
        plt.ylabel("speed [m/s]")
        plt.legend()
        plt.tight_layout(pad=0.5)
        plt.savefig("./data/test_speed.png")

        # 画间距
        plt.figure()
        sns.set(style="darkgrid", font_scale=1)
        min_index, min_value = -1, 100000000
        for i in range(len(X1_history)):
            dis.append(math.sqrt((X1_history[i] - X2_history[i])** 2 + (Y1_history[i] - Y2_history[i])** 2))

            if dis[-1] < min_value:
                min_value = dis[-1]
                min_index = i
        # plt.ylim([-5, 140])
        df = pd.DataFrame({'time': time, 'dis':dis})
        sns.lineplot(x="time", y="dis", data=df)
        plt.plot(time[min_index], dis[min_index], marker='o')
        plt.annotate("(%.1f,%.1f)" % (time[min_index], dis[min_index]), xy=(time[min_index], dis[min_index]), xytext=(10, -5), textcoords='offset points')
        plt.xlabel("time [s]")
        plt.ylabel("distance [m]")
        plt.tight_layout(pad=0.5)
        plt.savefig("./data/test_dis.png")

        # 画效用
        plt.figure()
        utility_1 = [-float(util) for util in utility_1]
        df = pd.DataFrame({'time': time, 'reward':utility_1})
        sns.lineplot(x="time", y="reward", data=df, legend='brief', label="vehicle 1")
        utility_2 = [-float(util) for util in utility_2]
        df = pd.DataFrame({'time': time, 'reward':utility_2})
        sns.lineplot(x="time", y="reward", data=df, legend='brief', label="vehicle 2")
        plt.xlabel("time [s]")
        plt.ylabel("reward")
        plt.legend()
        plt.savefig("./data/test_utility.png")



def get_parsers():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--buffer_size', default=200000, type=int)
    parser.add_argument('--agent_size', default=1, type=int)
    parser.add_argument('--max_iteration', default=100000)
    parser.add_argument('--lr_actor', default=0.0001, type=float)
    parser.add_argument('--lr_critic', default=0.001, type=float)
    parser.add_argument('--d_lim', default=15, type=float)
    parser.add_argument('--phi_lim', default=1.4, type=float)
    parser.add_argument('--u_lim', default=10.0, type=float)
    parser.add_argument('--deltaT', default=0.05, type=float)
    parser.add_argument('--gamma', default=0.98, type=float)
    parser.add_argument('--tra_pow', default=3, type=int)
    parser.add_argument('--mode', default="test",help="train/test")
    parser.add_argument('--step_number', default=1, type=int)
    parser.add_argument('--max_angle', default=0.15, type=float)
    parser.add_argument('--max_acc', default=2, type=float)
    parser.add_argument('--node_num', default=256, type=int)
    parser.add_argument('--dis_range', default=5, type=float)
    parser.add_argument('--load_old_network', default="True")
    parser.add_argument('--check_point', default=-1, type=int)

    return parser.parse_args()

def main():
    args = get_parsers()

    if args.mode == "train":
        writer = SummaryWriter()

        print('==== train ====')
        env = Dynamic(args)

        agent_1 = Agent(args, 1)
        agent_1.load_trajectory("./data/ref_traj_1")
        
        agent_2 = Agent(args, 2)
        agent_2.load_trajectory("./data/ref_traj_2")

        agent_1.tra_point_enemy = agent_2.tra_point
        agent_2.tra_point_enemy = agent_1.tra_point
        agent_1.v_x_enemy = agent_2.v_x
        agent_2.v_x_enemy = agent_1.v_x
        agent_1.kappa_enemy = agent_2.kappa
        agent_2.kappa_enemy = agent_1.kappa

        train = Train(args, agent_1, agent_2, env)
        train.save_args()

        iter_index = -1
        while iter_index <= args.max_iteration:
            iter_index += 1
            train.state_update(env, agent_1, agent_2)
            train.sample_from_buffer()
            value_loss_1, value_loss_2 = train.value_update(env, agent_1, agent_2)
            policy_loss_1, policy_loss_2 = train.policy_update(agent_1, agent_2)

            print("    s=[", end="")
            for i in range(14):
                print("%6.2f"%float(train.state[0, i]), end=",")
            print("]\tu1=[%6.2f, %6.2f]" % (float(train.agent_u1[0, 0]),float(train.agent_u1[0, 1])), end="")
            print("\tu2=[%6.2f, %6.2f]" % (float(train.agent_u2[0, 0]),float(train.agent_u2[0, 1])))
            if iter_index % 10 == 0:
                print("iter", iter_index, ":")
                print("    value_loss_1 = %.4f,\tpolicy_loss_1 = %.4f,\tvalue_loss_2 = %.4f,\tpolicy_loss_2 = %.4f" % (float(value_loss_1), float(policy_loss_1), float(value_loss_2), float(policy_loss_2)))
                
            if iter_index % 100 == 0:
                agent_1.save_network(iter_index)
                agent_2.save_network(iter_index)

                # agent_1.critic_scheduler.step()
                # agent_1.actor_scheduler.step()
                # agent_2.critic_scheduler.step()
                # agent_2.actor_scheduler.step()

            writer.add_scalars('value_loss', {'agent_1': value_loss_1, 'agent_2': value_loss_2}, global_step=iter_index)
            writer.add_scalars('policy_loss', {'agent_1': policy_loss_1, 'agent_2': policy_loss_2}, global_step=iter_index)

        writer.close()
        train.render()
    else:
        print('==== test ====')
        test = Test(args)
        test.run()

if __name__ == '__main__':
    main()