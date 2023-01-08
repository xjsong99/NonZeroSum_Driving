import numpy as np
import random
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything
seed_everything(0)

from dynamic import Dynamic
from agent import Agent, Vehicle
from train import Train
from test import Test

def get_parsers():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--buffer_size', default=200000, type=int)
    parser.add_argument('--round_size', default=10, type=int)
    parser.add_argument('--max_iteration', default=100000, type=int)
    parser.add_argument('--lr_actor', default=1e-3, type=float)
    parser.add_argument('--lr_critic', default=1e-3, type=float)
    parser.add_argument('--d_lim', default=10, type=float)
    parser.add_argument('--phi_lim', default=1.04, type=float)
    parser.add_argument('--u_lim', default=8, type=float)
    parser.add_argument('--deltaT', default=0.05, type=float)
    parser.add_argument('--gamma', default=0.98, type=float)
    parser.add_argument('--tra_pow', default=3, type=int)
    parser.add_argument('--mode', default="test",help="train/test")
    parser.add_argument('--step_number', default=20, type=int)
    parser.add_argument('--max_angle', default=0.15, type=float)
    parser.add_argument('--max_acc', default=2, type=float)
    parser.add_argument('--node_num', default=256, type=int)
    parser.add_argument('--dis_range', default=5, type=float)
    parser.add_argument('--time_headway', default=3, type=float)
    parser.add_argument('--load_old_network', default="False")
    parser.add_argument('--check_point', default=-1, type=int)

    return parser.parse_args()

def main():
    args = get_parsers()

    if args.mode == "train":
        writer = SummaryWriter()

        print('==== train ====')
        env = Dynamic(args)

        veh = []
        for i in range(4):
            veh.append(Vehicle(args, i))
            veh[i].load_trajectory("./data/ref_traj_"+str(i))

        agent = Agent(args)
        agent.train()

        train = Train(args, env, veh, agent)
        train.save_args()

        iter_index = -1
        while iter_index <= args.max_iteration:
            iter_index += 1
            train.state_update()
            train.sample_from_buffer()
            value_loss = train.value_update()
            policy_loss = train.policy_update()

            if iter_index % 10 == 0:
                print("iter", iter_index, ":")
                print("    value_loss = %.4f,\tpolicy_loss = %.4f" %\
                      (value_loss, policy_loss))
                
            if iter_index % 500 == 0:
                agent.save_network(iter_index)

            writer.add_scalar('value_loss', value_loss, global_step=iter_index)
            writer.add_scalar('policy_loss', policy_loss, global_step=iter_index)

        writer.close()
    else:
        print('==== test ====')
        test = Test(args)
        test.run()

if __name__ == '__main__':
    main()