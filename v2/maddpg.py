import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from memory import Memory
import model
from PIL import Image

# ！！！请务必在每次实验前更改exp_1名字，以记录实验于tensorboard中
TENSORBOARD_PATH = './logs/simple_1000_m1/'

class policy:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.cuda_avail = torch.cuda.is_available()
        self.actor = dict()
        self.actor_target = dict()
        self.critic = dict()
        self.critic_target = dict()
        self.actor_optim = dict()
        self.critic_optim = dict()
        for agent in env.world.agents:
            name = agent.name
            self.actor[name] = model.Actor(env, name)
            self.actor_target[name] = model.Actor(env, name)
            self.actor_target[name].load_state_dict(self.actor[name].state_dict())
            self.critic[name] = model.Critic(env)
            self.critic_target[name] = model.Critic(env)
            self.critic_target[name].load_state_dict(self.critic[name].state_dict())
            self.actor_optim[name] = optim.Adam(self.actor[name].parameters(), args.lr_a)
            self.critic_optim[name] = optim.Adam(self.critic[name].parameters(), args.lr_c)
            if self.cuda_avail:
                self.actor[name].cuda()
                self.actor_target[name].cuda()
                self.critic[name].cuda()
                self.critic_target[name].cuda()
        self.memory = Memory(env, args)

    def train(self):
        FloatTensor = torch.cuda.FloatTensor if self.cuda_avail else torch.FloatTensor
        if self.args.load:
            self.load()
        step = 0
        reward_history = []
        while step < self.args.num_epoch * self.args.num_step:
            epoch = step // self.args.num_step
            if step % self.args.num_step == 0:
                reward_history.append([])
                print(f'epoch {epoch + 1}')
                self.env.reset()
            if self.args.render:
                rgb_array = self.env.render()
                image = Image.fromarray(rgb_array)
                image.save('./img_test.png')
            self.interact(step, reward_history, epoch)
            if step >= self.args.random_step and \
                (step - self.args.random_step) % self.args.update_freq == 0:  #update network every update_freq
                with torch.no_grad():
                    batch = self.memory.sample(self.args.batch_size)
                    obs_shape, act_shape = 0, 0
                    for agent in self.env.world.agents:
                        name = agent.name
                        obs_shape += self.env.observation_spaces[name].shape[0]
                        act_shape += self.env.action_spaces[name].shape[0]
                    obs = np.ndarray((self.args.batch_size, obs_shape), dtype=np.float32)
                    action = np.ndarray((self.args.batch_size, act_shape), dtype=np.float32)
                    reward = np.ndarray((self.args.batch_size, 1), dtype=np.float32)
                    obs_new = np.ndarray((self.args.batch_size, obs_shape), dtype=np.float32)
                    action_new = np.ndarray((self.args.batch_size, act_shape), dtype=np.float32)
                    for i in range(self.args.batch_size):
                        obs[i] = np.concatenate([batch[name]['obs'][i] for name in batch])
                        action[i] = np.concatenate([batch[name]['action'][i] for name in batch])
                        reward[i][0] = batch[name]['reward'][i]
                        obs_new[i] = np.concatenate([batch[name]['obs_new'][i] for name in batch])
                        action_new[i] = np.concatenate([self.actor_target[name]\
                            (torch.tensor(batch[name]['obs_new'][i]).type(FloatTensor))\
                                .detach().type(FloatTensor).cpu() for name in batch])
                    obs = torch.tensor(obs).type(FloatTensor)
                    action = torch.tensor(action).type(FloatTensor)
                    reward = torch.tensor(reward).type(FloatTensor)
                    obs_new = torch.tensor(obs_new).type(FloatTensor)
                    action_new = torch.tensor(action_new).type(FloatTensor)
                for agent in self.env.world.agents:  #update network
                    name = agent.name
                    Q_new = self.critic_target[name](obs_new, action_new).detach().type(FloatTensor)
                    Q_target = (reward + self.args.gamma * Q_new).detach().type(FloatTensor)
                    Q = self.critic[name](obs, action).type(FloatTensor)
                    loss_c = nn.MSELoss()(Q, Q_target).mean().type(FloatTensor)
                    self.critic_optim[name].zero_grad()
                    loss_c.backward()
                    self.critic_optim[name].step()
                    loss_a = -self.critic[name](obs, action).mean().type(FloatTensor)
                    self.actor_optim[name].zero_grad()
                    loss_a.backward()
                    self.actor_optim[name].step()
                self.soft_update()#update parameter
            step += 1
        self.save()
        self.print_rew(reward_history)

    def test(self):
        FloatTensor = torch.cuda.FloatTensor if self.cuda_avail else torch.FloatTensor
        self.load()
        step = 0
        reward_history = []
        while step < self.args.test_epoch * self.args.test_step:
            epoch = step // self.args.test_step
            if step % self.args.test_step == 0:
                reward_history.append([])
                print(f'epoch {epoch + 1}')
                self.env.reset()
            if self.args.render:
                self.env.render()
            with torch.no_grad():
                for agent in self.env.world.agents:
                    obs = torch.from_numpy(self.env.observe(agent.name)).type(FloatTensor)
                    self.env.step(self.actor[agent.name](obs))
                reward_history[epoch].append(self.env._cumulative_rewards.copy())
            step += 1
        self.print_rew(reward_history)

    def interact(self, step, reward_history, epoch):
        FloatTensor = torch.cuda.FloatTensor if self.cuda_avail else torch.FloatTensor
        with torch.no_grad():
            for agent in self.env.world.agents:
                name = agent.name
                obs = torch.from_numpy(self.env.observe(name)).flatten().type(FloatTensor)
                if step < self.args.random_step: 
                    action = torch.from_numpy(np.random.uniform(0, 1, 5)).type(FloatTensor)
                else: action = self.actor[name](obs).detach().type(FloatTensor)
                self.env.step(action)
                self.memory.add(name, 'obs', obs)
                self.memory.add(name, 'action', action)
            rewards = self.env._cumulative_rewards.copy()
            reward_history[epoch].append(rewards)
            for name in rewards.keys():
                self.memory.add(name, 'reward', torch.tensor(rewards[name]).type(FloatTensor))
                obs_new = torch.from_numpy(self.env.observe(name)).flatten().type(FloatTensor)
                self.memory.add(name, 'obs_new', obs_new)
            self.memory.submit()
    
    def soft_update(self):
        for agent in self.env.world.agents:
            name = agent.name
            cur = self.critic[name].state_dict()
            tar = self.critic_target[name].state_dict()
            for key in cur.keys():
                tar[key] = self.args.tau * cur[key] + (1 - self.args.tau) * tar[key]
            self.critic_target[name].load_state_dict(tar)
            cur = self.actor[name].state_dict()
            tar = self.actor_target[name].state_dict()
            for key in cur.keys():
                tar[key] = self.args.tau * cur[key] + (1 - self.args.tau) * tar[key]
            self.actor_target[name].load_state_dict(tar)

    def load(self):
        for name in self.actor:
            self.actor[name].load_state_dict(torch.load(self.args.model_dir+f'/actor/{name}'))
            self.actor_target[name].load_state_dict(self.actor[name].state_dict())
            self.critic[name].load_state_dict(torch.load(self.args.model_dir+f'/critic/{name}'))
            self.critic_target[name].load_state_dict(self.critic[name].state_dict())

    def save(self):
        if not os.path.exists(self.args.model_dir):
            os.mkdir(self.args.model_dir)
        for dir in ['/actor/', '/critic/']:
            if not os.path.exists(self.args.model_dir+dir):
                os.mkdir(self.args.model_dir+dir)
        for name in self.actor:
            torch.save(self.actor[name].state_dict(), self.args.model_dir+f'/actor/{name}')
            torch.save(self.critic[name].state_dict(), self.args.model_dir+f'/critic/{name}')

    def tensorboard_w(self, reward_history:list): #print reward in tensorboard
        if not os.path.exists(TENSORBOARD_PATH):  # 如果路径不存在
            os.makedirs(TENSORBOARD_PATH)
        self.writer = SummaryWriter(TENSORBOARD_PATH)
        for epoch in range(len(reward_history)):
            item=reward_history[epoch][-1]
            for key in item.keys():
                tag0 = key
                scalar_value0 = item[key]
                self.writer.add_scalar(tag0, scalar_value0, epoch)
                
    def print_rew(self, reward_history):
        if not os.path.exists(self.args.log_dir):
            os.mkdir(self.args.log_dir)
        with open(self.args.log_dir+'/reward.log', 'w') as f:
            f.write(f'{reward_history}\n')
        self.tensorboard_w(reward_history)
