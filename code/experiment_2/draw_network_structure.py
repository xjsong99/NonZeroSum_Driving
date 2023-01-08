import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tensorwatch as tw
import torchvision.models

class Agent:
    def __init__(self, args):
        self.args = args

        self.actor_net = nn.Sequential(
            nn.Linear(12, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, 1),
            nn.Tanh()
        )

        self.critic_net = nn.Sequential(
            nn.Linear(12, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, self.args.node_num),
            nn.ELU(),
            nn.Linear(self.args.node_num, 12),
            nn.ReLU()
        )

parser = argparse.ArgumentParser()
parser.add_argument('--node_num', default=256, type=int)

agent = Agent(parser.parse_args())


dummy_input = torch.rand([1, 12])
writer = SummaryWriter()
writer.add_graph(agent.critic_net, (dummy_input,))
#writer.add_graph(agent.actor_net, (dummy_input,))


'''
img = tw.draw_model(agent.critic_net, torch.rand([1, 12]))
img.save("./data/net_structure.png")
'''