import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import math
import random
import pandas as pd
from PIL import Image
from dynamic import Dynamic
from agent import Agent, Vehicle


class Test:
    def __init__(self, args):
        self.args = args
        args.round_size = 1

        parameter_file = open("./data/test-parameter","w")
        parameter_file.write(self.args.__str__())
        parameter_file.close()
        
        self.env = Dynamic(args)

        self.veh = []
        for i in range(4):
            self.veh.append(Vehicle(args, i))
            self.veh[i].load_trajectory("./data/ref_traj_"+str(i))

        self.agent = Agent(args)
        self.agent.eval()

        # state = [X1, Y1, theta1, u1, v1, w1, X2, Y2, theta2, u2, v2, w2 \
        # index = [0,  1,  2,      3,  4,  5,  6,  7,  8,      9,  10, 11 \
        #          X3, Y3, theta3, u3, v3, w3, X4, Y4, theta4, u4, v4, w4 ]
        #          12, 13, 14,     15, 16, 17, 18, 19, 20,     21, 22, 23 ]

        self.state = torch.zeros([1, 4*6])
        self.state[0, 0] = -100 # X
        self.state[0, 1] = 0    # Y
        self.state[0, 6] = 15    # X
        self.state[0, 7] = -100 # Y
        self.state[0, 12] = 100 # X
        self.state[0, 13] = 20   # Y
        self.state[0, 18] = -5  # X
        self.state[0, 19] = 100 # Y
        for j in range(4):
            k = 6 * j
            self.state[0, 2+k] = self.veh[j].init_theta                                # theta
            self.state[0, 3+k] = random.uniform(1, 7)         # u
            self.state[0, 4+k] = random.uniform(-0.05, 0.05)                               # v
            self.state[0, 5+k] = random.uniform(-0.05, 0.05)                               # w
        self.state[0, 3] = 8
        self.state[0, 9] = 0
        self.state[0, 15] = 2
        self.state[0, 21] = 5
        self.state = self.state.cuda()

    def run(self):
        X_his = [[float(self.state[0, 0])],[float(self.state[0, 6])],[float(self.state[0, 12])],[float(self.state[0, 18])]]
        Y_his = [[float(self.state[0, 1])],[float(self.state[0, 7])],[float(self.state[0, 13])],[float(self.state[0, 19])]]
        speed_his = [[],[],[],[]]
        theta_his = [[],[],[],[]]
        utility_his = [[],[],[],[]]
        time = []

        iter_index = 0
        while True:
            if iter_index > 1000:
                break
            done = False
            for j in range(4):
                k = 6 * j
                if self.state[0, 0+k] < -100 or self.state[0, 0+k] > 100 or \
                   self.state[0, 1+k] < -100 or self.state[0, 1+k] > 100:
                        done = True
            if done == True:
                break
            if iter_index % 50 == 0:
                print("iter = ", iter_index)

            time.append(iter_index * self.args.deltaT)
            
            iter_index += 1
            
            u = [[],[],[],[]]
            for i in range(4):
                u[i] = self.agent.forward_actor(self.veh[i].state_transform(self.state))

            for i in range(4):
                utility_his[i].append(self.veh[i].utility(self.state, u[i]).detach().clone().cpu())
                speed_his[i].append(float(self.state[0, 3+6*i]))
                theta_his[i].append(float(self.state[0, 2+6*i]))
            
            u = torch.cat(u, 1)
            self.state = self.env.step(self.state, u)

            for j in range(4):
                k = 6 * j
                X_his[j].append(float(self.state[0, 0+k]))
                Y_his[j].append(float(self.state[0, 1+k]))
        
        for j in range(4):
            X_his[j].pop()
            Y_his[j].pop()

        # 画轨迹
        plot_reference = False # 指定是否画参考轨迹reference trajectory
        plt.figure()
        sns.set(style="dark")
        plt.xlim([-100, 100])
        plt.ylim([-100, 100])
        if plot_reference == True:
            tmp_x = [float(i[0, 0]) for i in self.veh[0].tra_point]
            tmp_y = [float(i[1, 0]) for i in self.veh[0].tra_point]
            plt.plot(tmp_x, tmp_y, linestyle='-', linewidth=2, color="green", label='reference')
            tmp_x = [float(i[0, 0]) for i in self.veh[1].tra_point]
            tmp_y = [float(i[1, 0]) for i in self.veh[1].tra_point]
            plt.plot(tmp_x, tmp_y, linestyle='-', linewidth=2, color="green")

        for i in range(4):
            plt.plot(X_his[i], Y_his[i], label='vehicle %d'%i)
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
        # plt.ylim([0,7])
        for i in range(4):
            df = pd.DataFrame({'time': time, 'speed':speed_his[i]})
            sns.lineplot(x="time", y="speed", data=df, legend='brief', label="vehicle %d"%i)
        plt.xlabel("time [s]")
        plt.ylabel("speed [m/s]")
        plt.legend()
        plt.tight_layout(pad=0.5)
        plt.savefig("./data/test_speed.png")
        '''
        # 画间距
        plt.figure()
        for i in range(4):
            for j in range(i):
                dis = []
                min_index, min_value = -1, 100000000
                for k in range(len(X_his[0])):
                    dis.append(math.sqrt((X_his[i][k] - X_his[j][k])**2 + (Y_his[i][k] - Y_his[j][k])**2))
                    if dis[-1] < min_value:
                        min_value = dis[-1]
                        min_index = k
                plt.plot(time, dis, linestyle='-')
                plt.plot(time[min_index], dis[min_index], marker='o')
                plt.annotate("(%.1f,%.1f)" % (time[min_index], dis[min_index]), xy=(time[min_index], dis[min_index]), xytext=(-20, -15), textcoords='offset points')
        plt.ylim([-5, 140])
        plt.xlabel("time [s]")
        plt.ylabel("distance [m]")
        plt.tight_layout(pad=0.5)
        plt.savefig("./data/test_dis.png")
        '''
        # 画reward
        plt.figure()
        for i in range(4):
            utility_his[i] = [-float(util) for util in utility_his[i]]
            df = pd.DataFrame({'time': time, 'reward':utility_his[i]})
            sns.lineplot(x="time", y="reward", data=df, legend='brief', label="vehicle %d"%i)
        plt.xlabel("time [s]")
        plt.ylabel("reward")
        plt.legend()
        plt.tight_layout(pad=0.5)
        plt.savefig("./data/test_utility.png")
        
        # 生成gif轨迹
        imgs = []
        color_set = ["blue","orange","green","red"]
        for t in range(0, 750):
            plt.figure()
            # sns.set(style="darkgrid", font_scale=1)
            # plt.xlim([-40, 40])
            # plt.ylim([-40, 40])
            for i in range(4):
                plt.plot(X_his[i][:t], Y_his[i][:t], c=color_set[i], label='vehicle %d'%i)
                if i == 0:
                    tail_center = (X_his[i][t], Y_his[i][t] - 1)
                elif i == 2:
                    tail_center = (X_his[i][t], Y_his[i][t] + 1)
                elif i == 1:
                    tail_center = (X_his[i][t] + 1, Y_his[i][t])
                elif i == 3:
                    tail_center = (X_his[i][t] - 1, Y_his[i][t])
                plt.gca().add_patch(plt.Rectangle(
                    xy=tail_center,
                    angle=theta_his[i][t]/3.15*180,
                    width=4,
                    height=2,
                    edgecolor=color_set[i],
                    fill=False,
                ))
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.legend()
            plt.tight_layout(pad=0.5)
            plt.savefig("./data/gif/"+"%d.png"%t)
            plt.close()
            imgs.append(Image.open("./data/gif/"+"%d.png"%t))
        imgs[0].save("./data/gif/traj.gif", save_all=True, append_images=imgs, duration=0.05)
        