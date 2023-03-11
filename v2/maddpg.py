import os
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from memory import Memory
import model
from PIL import Image

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
        self.total_step = 0
        path = self.args.tensorboard_dir + \
            (f'/test_{time.time()}' if self.args.test_mode \
             else f'/train_epoch{self.args.num_epoch}_step{self.args.num_step}_{time.time()}')
        if not os.path.exists(path): os.makedirs(path)
        self.writer = SummaryWriter(path)
        self.start_time = time.time()

    def run(self):
        if self.args.test_mode:
            self.load()
            for epoch in range(self.args.test_epoch):
                print(f'epoch {epoch + 1}')
                self.env.reset()
                rewards, _ = self.test(epoch)
                self.writer.add_scalars('test-rew-by-epoch', rewards, epoch)
        else:
            if self.args.load:
                self.load()
            for epoch in range(self.args.num_epoch):
                print(f'epoch {epoch + 1}')
                self.env.reset()
                self.train(epoch)
                if (epoch+1) % self.args.test_freq == 0:
                    self.save()
                    rewards = dict()
                    s_rewards = dict()
                    for te in range(self.args.test_epoch):
                        self.env.reset()
                        test_rewards, step_rewards = self.test(epoch+te)
                        for key in test_rewards:
                            if key in rewards: rewards[key] += test_rewards[key]
                            else: rewards[key] = test_rewards[key]
                        for k in step_rewards:
                            if k in s_rewards:
                                for key in step_rewards[k]:
                                    s_rewards[k][key] += step_rewards[k][key]
                            else: s_rewards[k] = step_rewards[k]
                    for key in rewards: rewards[key] /= self.args.test_epoch
                    self.writer.add_scalars('rew-by-epoch', rewards, epoch)
                    for k in s_rewards:
                        for key in s_rewards[k]:
                            s_rewards[k][key] /= self.args.test_epoch
                        self.writer.add_scalars('test-rew-by-step', s_rewards[k], self.total_step+k)
        self.writer.close()
    
    def train(self, epoch):
        rewards = dict()
        for step in range(self.args.num_step):
            if epoch % self.args.render_freq == 0 :
                self.render(epoch, step, 'train')
            self.interact(rewards)
            if self.total_step >= self.args.batch_size and \
                (self.total_step - self.args.batch_size) % self.args.update_freq == 0:  #update network every update_freq
                batch, obs, action, obs_new, action_new = self.sample()
                self.update(batch, obs, action, obs_new, action_new)
                self.soft_update()
            self.total_step += 1
        for key in rewards: rewards[key] /= self.args.num_step
        self.writer.add_scalars('rew-by-epoch', rewards, epoch)

    def test(self, epoch):
        FloatTensor = torch.cuda.FloatTensor if self.cuda_avail else torch.FloatTensor
        rewards = dict()
        step_rewards = dict()
        for step in range(self.args.num_step):
            if self.args.test_mode: self.render(epoch, step, 'test')
            with torch.no_grad():
                for agent in self.env.world.agents:
                    name = agent.name
                    obs = torch.from_numpy(self.env.observe(agent.name)).type(FloatTensor)
                    if self.args.env_name == 'simple':
                        action = self.action_select('net', name, obs)
                    elif self.args.env_name == 'tag':
                        if not agent.adversary:
                            action = self.action_select('net', name, obs)
                        else:
                            action = self.action_select('net', name, obs)
                            # action = self.action_select('chase', obs=obs)
                    elif self.args.env_name == 'spread':
                        action = self.action_select('net', name, obs)
                    self.env.step(action)
                for name in self.env.rewards:
                    key = f'test_{name}'
                    if key in rewards: rewards[key] += self.env.rewards[name]
                    else: rewards[key] = self.env.rewards[name]
                step_rewards[step] = self.env.rewards.copy()
            if self.args.test_mode: 
                self.writer.add_scalars('test-rew-by-step', self.env.rewards, self.total_step)
                self.total_step += 1
        for key in rewards: rewards[key] /= self.args.num_step
        return rewards, step_rewards

    def interact(self, rewards):
        FloatTensor = torch.cuda.FloatTensor if self.cuda_avail else torch.FloatTensor
        with torch.no_grad():
            for agent in self.env.world.agents:
                name = agent.name
                obs = torch.from_numpy(self.env.observe(name)).flatten().type(FloatTensor)
                action = self.action_select('random')
                if self.total_step >= self.args.random_step:
                    if self.args.env_name == 'simple':
                        self.args.noise *= self.args.decay
                        action = self.args.noise * action + \
                            (1 - self.args.noise) * self.action_select('net', name, obs)
                    if self.args.env_name == 'tag': #and not agent.adversary:
                        self.args.noise *= self.args.decay
                        action = self.args.noise * action + \
                            (1 - self.args.noise) * self.action_select('net', name, obs)
                # if self.args.env_name == 'tag' and agent.adversary:
                #     action = self.action_select('chase', obs=obs)
                self.env.step(action)
                self.memory.add(name, 'obs', obs)
                self.memory.add(name, 'action', action)
            for name in self.env.rewards:
                if name in rewards: rewards[name] += self.env.rewards[name]
                else: rewards[name] = self.env.rewards[name]
            self.writer.add_scalars('rew-by-step', self.env.rewards, self.total_step)
            for name in self.env.rewards:
                self.memory.add(name, 'reward', torch.tensor(self.env.rewards[name]).type(FloatTensor))
                obs_new = torch.from_numpy(self.env.observe(name)).flatten().type(FloatTensor)
                self.memory.add(name, 'obs_new', obs_new)
            self.memory.submit()
    
    def action_select(self, act_type, name=None, obs=None):
        FloatTensor = torch.cuda.FloatTensor if self.cuda_avail else torch.FloatTensor
        if act_type == 'net': return self.actor[name](obs).detach().type(FloatTensor)
        if act_type == 'chase':
            action = obs[2:4] - obs[12:14]
            action /= max(abs(action))
            action = [0, -min(0, action[0]), max(0, action[0]), -min(0, action[1]), max(0, action[1])]
            action = torch.tensor(action).type(FloatTensor)
            return action
        return torch.from_numpy(np.random.uniform(-1, 1, 5)).type(FloatTensor)

    def sample(self):
        with torch.no_grad():
            FloatTensor = torch.cuda.FloatTensor if self.cuda_avail else torch.FloatTensor
            batch = self.memory.sample(self.args.batch_size)
            obs_shape, act_shape = 0, 0

            for agent in self.env.world.agents:
                name = agent.name
                obs_shape += self.env.observation_spaces[name].shape[0]
                act_shape += self.env.action_spaces[name].shape[0]
            obs = np.ndarray((self.args.batch_size, obs_shape), dtype=np.float32)
            action = np.ndarray((self.args.batch_size, act_shape), dtype=np.float32)
            obs_new = np.ndarray((self.args.batch_size, obs_shape), dtype=np.float32)
            action_new = np.ndarray((self.args.batch_size, act_shape), dtype=np.float32)
            
            for i in range(self.args.batch_size):
                obs[i] = np.concatenate([batch[name]['obs'][i] for name in batch])
                action[i] = np.concatenate([batch[name]['action'][i] for name in batch])
                obs_new[i] = np.concatenate([batch[name]['obs_new'][i] for name in batch])
                action_new[i] = np.concatenate([self.actor_target[name]\
                    (torch.tensor(batch[name]['obs_new'][i]).type(FloatTensor))\
                        .detach().type(FloatTensor).cpu() for name in batch])
            obs = torch.tensor(obs).type(FloatTensor)
            action = torch.tensor(action).type(FloatTensor)
            obs_new = torch.tensor(obs_new).type(FloatTensor)
            action_new = torch.tensor(action_new).type(FloatTensor)
            
            return batch, obs, action, obs_new, action_new
    
    def update(self, batch, obs, action, obs_new, action_new):
        FloatTensor = torch.cuda.FloatTensor if self.cuda_avail else torch.FloatTensor
        act_shape = 0
        for agent in self.env.world.agents:
            name = agent.name
            if self.args.env_name == 'tag' and not agent.adversary:
                act_shape += self.env.action_spaces[name].shape[0]
                continue
            Q_new = self.critic_target[name](obs_new, action_new).detach().type(FloatTensor)
            Q_target = (torch.tensor(batch[name]['reward']).unsqueeze(1).type(FloatTensor) + self.args.gamma * Q_new).type(FloatTensor)
            Q = self.critic[name](obs, action).type(FloatTensor)
            loss_c = nn.MSELoss()(Q, Q_target).type(FloatTensor)
            self.critic_optim[name].zero_grad()
            loss_c.backward()

            self.critic_optim[name].step()
            old_action = torch.tensor(np.copy(action.cpu())).type(FloatTensor)
            old_action[:, act_shape:act_shape+self.env.action_spaces[name].shape[0]] = \
                self.actor[name](torch.tensor(batch[name]['obs']).type(FloatTensor))
            act_shape += self.env.action_spaces[name].shape[0]
            loss_a = -self.critic[name](obs, old_action).mean().type(FloatTensor)
            self.actor_optim[name].zero_grad()
            loss_a.backward()

            self.actor_optim[name].step()
    
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

    def render(self, epoch, step, mode):
        if self.args.render_mode is None: return
        if not os.path.exists(f'{self.args.image_dir}/{self.start_time}'):
            os.mkdir(f'{self.args.image_dir}/{self.start_time}')
        if self.args.render_mode == 'rgb_array':
            self.env.draw()
            rgb_array = self.env.render()
            image = Image.fromarray(rgb_array)
            image.save(f'{self.args.image_dir}/{self.start_time}/img_{epoch+1}_{step+1}.png')
            if step+1 == self.args.num_step:
                frames = []
                for i in range(self.args.num_step):
                    frames.append(Image.open(f'{self.args.image_dir}/{self.start_time}/img_{epoch+1}_{i+1}.png'))
                    os.remove(f'{self.args.image_dir}/{self.start_time}/img_{epoch+1}_{i+1}.png')
                frames[0].save(f'{self.args.image_dir}/{self.start_time}/{mode}_gif_{epoch+1}.gif', save_all=True, \
                               append_images=frames[1:], duration=1000/self.args.fps)
    
    def load(self):
        for name in self.actor:
            self.actor[name].load_state_dict(torch.load(self.args.model_dir+f'/actor/{name}'))
            self.actor_target[name].load_state_dict(self.actor[name].state_dict())
            self.critic[name].load_state_dict(torch.load(self.args.model_dir+f'/critic/{name}'))
            self.critic_target[name].load_state_dict(self.critic[name].state_dict())

    def save(self):
        for name in self.actor:
            torch.save(self.actor[name].state_dict(), self.args.model_dir+f'/actor/{name}')
            torch.save(self.critic[name].state_dict(), self.args.model_dir+f'/critic/{name}')
