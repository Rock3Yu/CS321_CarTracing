import argparse
import os
from typing import Optional, Tuple
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (BasePolicy, DDPGPolicy, MultiAgentPolicyManager,)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
import envs.simple_spread_v2 as spread
import argument
from model import Actor, Critic

"""当step per epoch是step per collect的整数倍时,total_step=step per epoch*maxepoch
当不是整数倍时, n=ceil(step per epoch/step per collect),total_step=n*step per collect* maxepoch
"""


class Tian_adapt:
    def __init__(self):
        args = argument.argparse()


def init_env():
    args = argument.get_args()
    env_ori = spread.env(N=args.num_agent, continuous_actions=True,
                         render_mode='rgb_array', local_ratio=args.local_ratio, args=args).env.env
    return env_ori


def get_env_fn():
    args = argument.get_args()
    env_ori = spread.env(N=args.num_agent, continuous_actions=True,
                         render_mode='rgb_array', local_ratio=args.local_ratio, args=args).env.env
    return PettingZooEnv(env_ori)


def get_agents(
    args: argparse.Namespace = argument.get_args(),
    agent_learn: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    test=True
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = init_env()
    agents = []
    for i in range(args.num_agent):
        # model
        actor = Actor(env, f"agent_{i}")
        critic = Critic(env, f"agent_{i}")
        if optim is None:
            a_optim = torch.optim.Adam(actor.parameters(), lr=args.lr_a)
            c_optim = torch.optim.Adam(critic.parameters(), lr=args.lr_c)
        agent_learn = DDPGPolicy(
            actor,
            a_optim,
            critic,
            c_optim,
            args.lr_a,
            args.gamma
        )
        if args.load:  # load in file
            agent_learn.load_state_dict(torch.load(args.model_dir))
        agents.append(agent_learn)
    policy = MultiAgentPolicyManager(agents, PettingZooEnv(env))
    return policy, agents


def train_agent(
    args: argparse.Namespace = argument.get_args(),
    agent_learn: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:

    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env_fn])
    test_envs = DummyVectorEnv([get_env_fn])
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)  # make a seed
        train_envs.seed(args.seed)
        test_envs.seed(args.seed)

    # ======== agent setup =========
    policy, agents = get_agents(
        args, agent_learn=agent_learn, optim=optim
    )

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.memory_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_episode=args.explore_epoch)
    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.log_dir, str(time.time()))

    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))  # record the info of experiment
    logger = TensorboardLogger(writer)

    # ======== callback functions used during training =========
    model_save_path = f"{args.model_dir}/{int(time.time())}"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    model_save_path = model_save_path+"/actor.pth"

    def save_best_fn(policy):
        torch.save(
            agents[0], model_save_path)

    def stop_fn(mean_rewards):  # ：停止条件，输入是当前平均总奖励回报（the average undiscounted returns），返回是否要停止训练
        # return mean_rewards >= args.win_rate
        return False  # not stop

    def train_fn(epoch, env_step):  # 在每个epoch训练之前被调用的函数，输入的是当前第几轮epoch和当前用于训练的env一共step了多少次。
        # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)
        pass

    def test_fn(epoch, env_step):  # 在每个epoch测试之前被调用的函数，输入的是当前第几轮epoch和当前用于训练的env一共step了多少次。
        # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)
        pass

    result = offpolicy_trainer(
        policy,  # policy
        train_collector,  # train_collector
        test_collector,  # test_collector
        args.num_epoch,  # 最大允许的训练轮数
        args.num_step,  # 总共要更新多少次网络
        args.num_step-20,  # 每次更新前要收集多少帧与环境的交互数据。上面的代码参数意思是，每收集10帧进行一次网络更新
        episode_per_test=1,  # episode_per_test test_num
        batch_size=args.batch_size,
        # train_fn=train_fn,
        # test_fn=test_fn,
        # stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False
    )

    return result, agents[0]

# ======== a test function that tests a pre-trained agent ======


def watch(
    args: argparse.Namespace = argument.get_args(),
    agent_learn: Optional[BasePolicy] = None,
) -> None:
    env = DummyVectorEnv([get_env_fn])
    policy, agents = get_agents(
        args, agent_learn=agent_learn
    )
    policy.eval()
    # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(
        n_episode=args.test_epoch, render=args.render_mode)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")


def main(args):
    if args.test_mode == True:
        policy, agents = get_agents(args)
        watch(args, agents)
    else:
        result, agents = train_agent(args)
        watch(args, agents)


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = argument.get_args()
    result, agents = train_agent(args)
