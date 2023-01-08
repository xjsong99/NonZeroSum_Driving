import torch
import torch.nn as nn
import torch.optim as optim
import os
import math

class Vehicle:
    def __init__(self, args, vehicle_type):
        self.args = args

        # 0:西->东  1:南->北  2:东->西  3:北->南
        self.vehicle_type = vehicle_type

        #要跟踪的大地坐标点
        # [tensor[X0,Y0],tensor[X1,Y1],...]
        self.tra_point = []

        # 可行驾驶区域
        # [x_min, x_max, y_min, y_max] 
        self.drive_range_xmin = 0
        self.drive_range_xmax = 0
        self.drive_range_ymin = 0
        self.drive_range_ymax = 0

        # 初始车头朝向
        if vehicle_type == 0:
            self.init_theta = 0
        elif vehicle_type == 1:
            self.init_theta = math.pi/2
        elif vehicle_type == 2:
            self.init_theta = math.pi
        else:
            self.init_theta = math.pi*3/2

        self.v_x_desired = 0

        # utility参数
        self.c_d, self.c_phi, self.c_vx = 1.5 / (self.args.d_lim ** 2), 0.1 / (self.args.phi_lim ** 2), 0.8 / ((self.args.u_lim / 2)** 2)
        self.c_delta, self.c_acc = 0.001 / (self.args.max_angle ** 2), 0.3 / (self.args.max_acc ** 2)
        self.c_thw = 0 # 0.1 / self.args.time_headway
        self.c_dis = 4.5 / (self.args.dis_range ** 2)

    def state_transform(self, state_global):
        '''
        <input>
        state_global : tensor[round_size, 4*6]
        <return>
        state_veh    : tensor[round_size, 17]
        '''
        # state_veh = [d1, dtheta1, u1, v1, w1, X2_veh, Y2_veh, theta2, u2, X3_veh, Y3_veh, theta3, u3, X4_veh, Y4_veh, theta4, u4]
        # index     = [0,  1,       2,  3,  4,  5,      6,      7,      8,  9,      10,     11,     12, 13,     14,     15,     16]
        
        state_veh = []
        for i in range(17):
            state_veh.append([])
        
        veh_index = 6 * self.vehicle_type
        if self.vehicle_type == 0:
            state_veh[0] = state_global[:, 1 + veh_index]
        elif self.vehicle_type == 2:
            state_veh[0] = - state_global[:, 1 + veh_index]
        elif self.vehicle_type == 1:
            state_veh[0] = - state_global[:, 0 + veh_index]
        else:
            state_veh[0] = state_global[:, 0 + veh_index]
        state_veh[1] = state_global[:, 2 + veh_index] - self.init_theta
        state_veh[2] = state_global[:, 3 + veh_index]
        state_veh[3] = state_global[:, 4 + veh_index]
        state_veh[4] = state_global[:, 5 + veh_index]
        
        for i in range(3):
            j = 4 * i
            state_veh[5+j] = torch.cos(state_global[:, 2 + veh_index]) * (state_global[:, (0 + veh_index + 6*(i+1))%24] - state_global[:, 0 + veh_index]) \
                           + torch.sin(state_global[:, 2 + veh_index]) * (state_global[:, (1 + veh_index + 6*(i+1))%24] - state_global[:, 1 + veh_index])
            state_veh[6+j] = -torch.sin(state_global[:, 2 + veh_index]) * (state_global[:, (0 + veh_index + 6*(i+1))%24] - state_global[:, 0 + veh_index]) \
                           + torch.cos(state_global[:, 2 + veh_index]) * (state_global[:, (1 + veh_index + 6*(i+1))%24] - state_global[:, 1 + veh_index])
            state_veh[7+j] = torch.remainder(state_global[:, (2 + veh_index + 6*(i+1))%24] - state_global[:, 2 + veh_index], 2*math.pi)
            state_veh[8+j] = state_global[:, (3 + veh_index + 6*(i+1))%24]

        for i in range(17):
            state_veh[i] = state_veh[i].unsqueeze(1)

        return torch.cat(state_veh, 1)
    
    def utility(self, state_global, u):
        '''
        <input>
        state_global : tensor[round_size, 4*6]
        u            : tensor[round_size, 2]
        <return>
        l            : tensor[round_size, 1]
        '''
        # state_veh = [d1, dtheta1, u1, v1, w1, X2_veh, Y2_veh, theta2, u2, X3_veh, Y3_veh, theta3, u3, X4_veh, Y4_veh, theta4, u4]
        # index     = [0,  1,       2,  3,  4,  5,      6,      7,      8,  9,      10,     11,     12, 13,     14,     15,     16]

        # u     = [a, δ]
        # index = [0, 1]

        state_veh = self.state_transform(state_global)

        l = (self.c_d * torch.pow(state_veh[:, 0], 2)).unsqueeze(1) + \
            (self.c_phi * torch.pow(state_veh[:, 1], 2)).unsqueeze(1) + \
            (self.c_vx * torch.pow(state_veh[:, 2] - self.v_x_desired, 2)).unsqueeze(1) + \
            self.c_delta * torch.pow(u[:, 0], 2) + \
            self.c_acc * torch.pow(u[:, 1], 2)
        # # 学时距
        # for i in range(3):
        #     j = 4 * i
        #     thw_value = self.args.time_headway - (torch.sqrt(torch.pow(state_veh[:, 5+j], 2) + torch.pow(state_veh[:, 6+j], 2))/state_global[:, 3+6*i]).unsqueeze(1)
        #     thw_value = torch.where(thw_value < 0, torch.zeros([1,1]).cuda(), thw_value)  # thw_value[thw_value < 0] = 0
        #     l = l + self.c_thw * thw_value
        # 学距离
        for i in range(3):
            j = 4 * i
            D_value = self.args.dis_range**2 - (torch.pow(state_veh[:, 5+j], 2) + torch.pow(state_veh[:, 6+j], 2)).unsqueeze(1)
            D_value = torch.where(D_value < 0, torch.zeros([1,1]).cuda(), D_value)  # D_value[D_value < 0] = 0
            l = l + self.c_dis * D_value

        return l*0.001

    def load_trajectory(self, path):
        file_trajectory = open(path, "r")
        data = file_trajectory.readlines()
        for i in range(len(data) - 1):
            point = data[i].split()
            point_tensor = torch.zeros([2, 1])
            point_tensor[0, 0], point_tensor[1, 0] = float(point[0]), float(point[1])
            self.tra_point.append(point_tensor)
        self.v_x_desired = float(data[-1])
        if self.vehicle_type == 0 or self.vehicle_type == 2:
            self.drive_range_xmin = min(self.tra_point[0][0,0],self.tra_point[-1][0,0])
            self.drive_range_xmax = max(self.tra_point[0][0,0],self.tra_point[-1][0,0])
            self.drive_range_ymin = -self.args.d_lim
            self.drive_range_ymax = self.args.d_lim
        else:
            self.drive_range_xmin = -self.args.d_lim
            self.drive_range_xmax = self.args.d_lim
            self.drive_range_ymin = min(self.tra_point[0][1,0],self.tra_point[-1][1,0])
            self.drive_range_ymax = max(self.tra_point[0][1,0],self.tra_point[-1][1,0])

class Agent(nn.Module):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.args = args

        self.norm_matrix = torch.tensor([1.0 / self.args.d_lim, 1.0 / self.args.phi_lim, 1.0 / self.args.u_lim, 1.0, 1.0, \
                                         1.0 / 100, 1.0 / 100, 1.0 / (2*3.14), 1.0 / self.args.u_lim, \
                                         1.0 / 100, 1.0 / 100, 1.0 / (2*3.14), 1.0 / self.args.u_lim, \
                                         1.0 / 100, 1.0 / 100, 1.0 / (2*3.14), 1.0 / self.args.u_lim  ], dtype=torch.float32).cuda()
        self.gain_matrix = torch.tensor([[self.args.max_acc, self.args.max_angle]]).cuda()

        self.actor_net = nn.Sequential(
            nn.Linear(17, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, 2),
            nn.Tanh()
        ).cuda()
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), self.args.lr_actor)

        for m in self.actor_net.modules():
            if isinstance(m, nn.Linear):
                # torch.nn.init.uniform_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)

        self.critic_net = nn.Sequential(
            nn.Linear(17, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, 1),
            nn.ReLU(),
        ).cuda()
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), self.args.lr_critic)
        
        for m in self.critic_net.modules():
            if isinstance(m, nn.Linear):
                # torch.nn.init.uniform_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)

        if (args.mode=="train" and args.load_old_network == "True") \
           or args.mode=="test":
            self.load_network(args.check_point)

    def forward_actor(self, x):
        '''
        <input>
        state_veh    : tensor[batch_size, 17]
        <return>
        u            : tensor[batch_size, 2]
        '''
        x = torch.mul(x, self.norm_matrix)
        x = self.gain_matrix * self.actor_net(x)
        return x
        # action_noise = torch.normal(0,torch.tensor([[0.15, 0.03]]).repeat(x.shape[0],1)).cuda()
        # return x + action_noise

    def forward_critic(self, x):
        '''
        <input>
        state_veh    : tensor[batch_size, 17]
        <return>
        v(x)         : tensor[batch_size, 1]
        '''
        x = torch.mul(x, self.norm_matrix)
        x = self.critic_net(x)
        return x

    def save_network(self, iter_index):
        path = "./data"+"/actor_iter_" + str(iter_index)
        torch.save(self.actor_net.state_dict(), path)
        path = "./data"+"/critic_iter_" + str(iter_index)
        torch.save(self.critic_net.state_dict(), path)
    
    def load_network(self, check_point_index):
        iter_index = 0
        if check_point_index == -1:
            while os.path.isfile("./data/"+"/actor_iter_" + str(iter_index + 1000)):
                iter_index += 1000
        else:
            iter_index = check_point_index
        print("load data from iter_index = ", iter_index)
        path = "./data"+"/actor_iter_" + str(iter_index)
        self.actor_net.load_state_dict(torch.load(path))
        path = "./data"+"/critic_iter_" + str(iter_index)
        self.critic_net.load_state_dict(torch.load(path))
    