import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DDPGPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
import envs.simple_spread_v2 as spread
from envs.simple_env import make_env
import argument
from model import Actor ,Critic

def init_env():
    args= argument.get_args()
    env_ori = spread.env(N=args.num_agent,continuous_actions = True, render_mode = 'rgb_array', local_ratio = args.local_ratio).env.env
    return env_ori

def get_env_fn():
    args= argument.get_args()
    env_ori = spread.env(N=args.num_agent,continuous_actions = True, render_mode = 'rgb_array', local_ratio = args.local_ratio).env.env
    return PettingZooEnv(env_ori)

def get_agents(
    args: argparse.Namespace = argument.get_args(),
    agent_learn: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = init_env()
    agents=[]
    for i in range(args.num_agent):
        # model
        actor=Actor(env,f"agent_{i}")
        critic=Critic(env,f"agent_{i}")
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
        if args.load:#load in file
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
    train_envs = DummyVectorEnv([ get_env_fn])
    test_envs = DummyVectorEnv([get_env_fn])
    # seed
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed) #make a seed
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
    log_path = os.path.join(args.log_dir, 'spread')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args)) #record the info of experiment
    logger = TensorboardLogger(writer)



    #todo: Configure the env to spread:
    # ======== callback functions used during training =========
    def save_best_fn(policy):
        if hasattr(args, 'model_dir'):
            model_save_path = os.path.join(args.model_dir)
            torch.save(
                agents[0], model_save_path )

    def stop_fn(mean_rewards):#：停止条件，输入是当前平均总奖励回报（the average undiscounted returns），返回是否要停止训练
        # return mean_rewards >= args.win_rate
        return False #not stop

    def train_fn(epoch, env_step):# 在每个epoch训练之前被调用的函数，输入的是当前第几轮epoch和当前用于训练的env一共step了多少次。
        # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)
        pass

    def test_fn(epoch, env_step):#在每个epoch测试之前被调用的函数，输入的是当前第几轮epoch和当前用于训练的env一共step了多少次。
        # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)
        pass

    result = offpolicy_trainer(
        policy, #policy
        train_collector, #train_collector
        test_collector, #test_collector
        args.num_epoch, #epoch
        args.num_step, #step per epoch
        args.explore_epoch, # 网络更新之前收集的帧数
        episode_per_test= 1, #episode_per_test test_num
        batch_size=args.batch_size,
        # train_fn=train_fn,
        # test_fn=test_fn,
        # stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=True
    )

    return result, policy.policies[agents[0]]

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
    result = collector.collect(n_episode=1, render=args.render_mode)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = argument.get_args()
    result, agent = train_agent(args)
    watch(args, agent)
