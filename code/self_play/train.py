import torch
from buffer import Buffer
import matplotlib.pyplot as plt

class Train:
    def __init__(self, args, env, veh, agent):
        self.args = args
        self.env = env
        self.veh = veh
        self.agent = agent

        self.buffer = Buffer(args)
        self.mini_batch = []

        self.state = torch.zeros([self.args.round_size, 4*6])
        for i in range(self.args.round_size):
            env.init_state(self.state, veh, i)

        self.state.detach_()
        self.state = self.state.cuda()
        
        self.mini_batch_forward = []
        self.u_forward = []
        self.l_forward = [[],[],[],[]]
        for i in range(args.step_number):
            self.mini_batch_forward.append([])
            self.u_forward.append([[],[],[],[]])
            for j in range(len(self.veh)):
                self.l_forward[j].append([])
        self.mini_batch_forward.append([])
        self.l_sum = [[],[],[],[]]

    def save_args(self):
        parameter_file = open("./data/train-parameter","w")
        parameter_file.write(self.args.__str__())
        parameter_file.close()
    
    def state_update(self):
        # self.draw_state(self.state_1, self.state_1_forward, agent)

        # 计算控制量
        u = [[],[],[],[]]
        for i in range(len(self.veh)):
            u[i] = self.agent.forward_actor(self.veh[i].state_transform(self.state))
        u = torch.cat(u, 1)

        # 计算下一状态state_next
        state_next = self.env.step(self.state, u)        

        self.state = state_next.clone().detach()
        self.state = self.env.check_done(self.state, self.veh)

        self.buffer.add(self.state)
        
    def sample_from_buffer(self):
        self.mini_batch = self.buffer.sample()

    def value_update(self):
        if len(self.mini_batch) == 0:
            return 0

        # 计算J(x(0))
        J_current = [[],[],[],[]]
        for i in range(len(self.veh)):
            J_current[i] = self.agent.forward_critic(self.veh[i].state_transform(self.mini_batch).detach())

        # 计算l_sum=l(x(0))+γl(x(1))+...+γ^(step_number-1)*l(x(0+step_number-1))
        self.mini_batch_forward[0] = self.mini_batch.detach()
        gamma_now = self.args.gamma
        for i in range(self.args.step_number):
            for j in range(len(self.veh)):
                self.u_forward[i][j] = self.agent.forward_actor(self.veh[j].state_transform(self.mini_batch_forward[i]))
                self.l_forward[j][i] = gamma_now * self.veh[j].utility(self.mini_batch_forward[i], self.u_forward[i][j])
            self.mini_batch_forward[i + 1] = self.env.step(self.mini_batch_forward[i], torch.cat(self.u_forward[i],1))
            gamma_now = gamma_now * self.args.gamma

        for j in range(len(self.veh)):
            self.l_sum[j] = torch.sum(torch.cat(self.l_forward[j], 1), 1)

        # 计算J(x(0+step_number))
        J_step_number = [[],[],[],[]]
        for j in range(len(self.veh)):
            J_step_number[j] = self.agent.forward_critic(self.veh[j].state_transform(self.mini_batch_forward[-1]))

        target_value = [[],[],[],[]]
        for j in range(len(self.veh)):
            target_value[j] = self.l_sum[j].detach() + gamma_now * J_step_number[j].detach()

        value_loss = 1 / 2 * ( \
                     torch.mean(torch.pow(target_value[0] - J_current[0], 2)) + \
                     torch.mean(torch.pow(target_value[1] - J_current[1], 2)) + \
                     torch.mean(torch.pow(target_value[2] - J_current[2], 2)) + \
                     torch.mean(torch.pow(target_value[3] - J_current[3], 2))   \
        )

        self.agent.critic_optimizer.zero_grad()
        value_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic_net.parameters(), 0.1)
        self.agent.critic_optimizer.step()
        # agent.critic_scheduler.step()
        
        return float(value_loss)
        
    def policy_update(self):
        if len(self.mini_batch) == 0:
            return 0

        J_step_number = [[],[],[],[]]
        for j in range(len(self.veh)):
            J_step_number[j] = self.agent.forward_critic(self.veh[j].state_transform(self.mini_batch_forward[-1]))

        policy_loss = torch.mean(self.l_sum[0] + J_step_number[0]) + \
                      torch.mean(self.l_sum[1] + J_step_number[1]) + \
                      torch.mean(self.l_sum[2] + J_step_number[2]) + \
                      torch.mean(self.l_sum[3] + J_step_number[3])

        self.agent.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.agent.actor_optimizer.step()

        # torch.nn.utils.clip_grad_norm_(agent.actor_net.parameters(), 0.1)
        # agent.actor_scheduler.step()
        
        return float(policy_loss)

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