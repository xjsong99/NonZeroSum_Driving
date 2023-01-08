import torch
import random

class Dynamic:
    def __init__(self, args):
        # state = [X1, Y1, e_d, e_phi, r, v_y, X2, Y2]
        self.args = args
        self.deltaT = args.deltaT
        self.kf, self.kr = -88000., -94000.
        self.lf, self.lr = 1.14, 1.4
        self.m = 1500.
        self.Iz = 2420.

    def step(self, state, u):
        '''
        <input>
        state     :tensor[round_size, 4*6]
        u         :tensor[round_size, 4*2]
        <return>
        state     :tensor[round_size, 4*6]
        '''
        # state = [X1, Y1, theta1, u1, v1, w1, X2, Y2, theta2, u2, v2, w2 \
        # index = [0,  1,  2,      3,  4,  5,  6,  7,  8,      9,  10, 11 \
        #          X3, Y3, theta3, u3, v3, w3, X4, Y4, theta4, u4, v4, w4 ]
        #          12, 13, 14,     15, 16, 17, 18, 19, 20,     21, 22, 23 ]

        # u     = [a, δ]
        # index = [0, 1]

        # 更新状态state
        new_state = []
        for i in range(state.shape[1]):
            new_state.append([])

        for i in range(4):
            j = i * 6
            new_state[0+j] = state[:, 0+j] + self.args.deltaT * (state[:, 3+j] * torch.cos(state[:, 2+j]) - state[:, 4+j] * torch.sin(state[:, 2+j]))
            new_state[1+j] = state[:, 1+j] + self.args.deltaT * (state[:, 3+j] * torch.sin(state[:, 2+j]) + state[:, 4+j] * torch.cos(state[:, 2+j]))
            new_state[2+j] = state[:, 2+j] + self.args.deltaT * state[:, 5+j]
            new_state[3+j] = state[:, 3+j] + self.args.deltaT * u[:, 0+2*i]
            new_state[4+j] = (self.m * state[:, 3+j] * state[:, 4+j] + self.args.deltaT * (self.lf * self.kf - self.lr * self.kr) * state[:, 5+j] - self.args.deltaT * self.kf * u[:, 1+2*i] * state[:, 3+j] - self.args.deltaT * self.m * torch.pow(state[:, 3+j], 2) * state[:, 5+j]) \
                        / (self.m * state[:, 3+j] - self.deltaT * (self.kf + self.kr))
            new_state[5+j] = (self.Iz * state[:, 3+j] * state[:, 5+j] + self.deltaT * (self.lf * self.kf - self.lr * self.kr) * state[:, 4+j] - self.deltaT * self.lf * self.kf * u[:, 1+2*i] * state[:, 3+j])\
                        / (self.Iz * state[:, 3+j] - self.args.deltaT * (self.lf * self.lf * self.kf + self.lr * self.lr * self.kr))

        for i in range(state.shape[1]):
            new_state[i] = new_state[i].unsqueeze(1)

        return torch.cat(new_state, 1)
    
    def init_state(self, state, veh, i):
        for j in range(4):
            k = 6 * j
            rand_index = random.randint(0, len(veh[j].tra_point)-1)
            state[i, 0+k] = veh[j].tra_point[rand_index][0, 0] + 2 * torch.normal(torch.tensor(0.0)) # X
            state[i, 1+k] = veh[j].tra_point[rand_index][1, 0] + 2 * torch.normal(torch.tensor(0.0)) # Y
            state[i, 2+k] = veh[j].init_theta + random.uniform(-0.05, 0.05) # theta
            state[i, 3+k] = veh[j].v_x_desired + random.uniform(-1, 1)  # u
            state[i, 4+k] = random.uniform(-0.05, 0.05)  # v
            state[i, 5+k] = random.uniform(-0.05, 0.05)  # w

    def check_done(self, state, veh):
        for i in range(self.args.round_size):
            done = False
            for j in range(4):
                k = 6 * j
                if state[i, 0+k] < veh[j].drive_range_xmin or state[i, 0+k] > veh[j].drive_range_xmax or \
                   state[i, 1+k] < veh[j].drive_range_ymin or state[i, 1+k] > veh[j].drive_range_ymax or \
                   torch.abs(state[i, 2+k] - veh[j].init_theta) > self.args.phi_lim or \
                   state[i, 3+k] < 0 or \
                   state[i, 3+k] > self.args.u_lim:
                        done = True
            if done == True:
                # print("【check done】")
                # print("    s=[", end="")
                # for j in range(24):
                #     print("%6.2f"%float(state[i, j]), end=",")
                # print("]")
                # for j in range(4):
                #     k = 6 * j
                #     print(state[i, 0+k] < veh[j].drive_range_xmin , state[i, 0+k] > veh[j].drive_range_xmax , \
                #     state[i, 1+k] < veh[j].drive_range_ymin , state[i, 1+k] > veh[j].drive_range_ymax , \
                #     torch.abs(state[i, 2+k] - veh[j].init_theta) > self.args.phi_lim , \
                #     state[i, 3+k] < 0 , \
                #     state[i, 3+k] > self.args.u_lim)
                self.init_state(state, veh, i)
        return state