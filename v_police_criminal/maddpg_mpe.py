import os
import shutil
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
        self.start_time = time.time()
        self.env = env
        self.args = args
        self.args.log_dir += f'/{self.start_time}'
        self.args.image_dir += f'/{self.start_time}'
        if not os.path.exists(self.args.model_dir): os.makedirs(self.args.model_dir)
        if not os.path.exists(self.args.log_dir): os.makedirs(self.args.log_dir)
        if not os.path.exists(self.args.image_dir): os.makedirs(self.args.image_dir)
        self.cuda_avail = torch.cuda.is_available()
        self.actor = {}
        self.actor_target = {}
        self.critic = {}
        self.critic_target = {}
        self.actor_optim = {}
        self.critic_optim = {}
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
        self.FT = torch.cuda.FloatTensor if self.cuda_avail else torch.FloatTensor
        self.total_epoch = 1
        self.temp_rewards = {}
        self.writer = SummaryWriter(self.args.log_dir)
        with open(f'{self.args.log_dir}/benchmark.csv', 'w') as f:
            f.write('epoch,step,agent,data\n')
        self.writer.add_text("args", str(args)) #record the hyperparameters

    def run(self):
        if self.args.test_mode:
            self.load()
            while self.total_epoch <= self.args.test_epoch:
                print(f'epoch {self.total_epoch}')
                self.env.reset(self.args.seed)
                if self.total_epoch % self.args.render_freq == 0: self.test(True, self.total_epoch)
                else: self.test(False, self.total_epoch)
                self.total_epoch += 1
        else:
            if self.args.load: self.load()
            self.explore()
            while self.total_epoch <= self.args.num_epoch:
                print(f'epoch {self.total_epoch}')
                self.env.reset(self.args.seed)
                if self.total_epoch % self.args.render_freq == 0: self.train(True)
                else: self.train(False)
                self.total_epoch += 1
                if self.total_epoch % self.args.save_freq == 0: self.save()
                if self.total_epoch % self.args.test_freq == 0: 
                    for epoch in range(self.args.test_epoch):
                        self.env.reset(self.args.seed)
                        self.test(False, epoch+1+self.total_epoch)
            if self.total_epoch % self.args.save_freq != 0: self.save()
        self.writer.close()
    
    def explore(self):
        for _ in range(self.args.explore_epoch):
            self.env.reset(self.args.seed)
            for _ in range(self.args.num_step):
                self.interact(False, 'random', None, None, 'train')

    def train(self, render):
        self.temp_rewards = {}
        for step in range(self.args.num_step):
            self.interact(render, 'net', self.total_epoch, step, 'train')
            for agent in self.env.world.agents: self.benchmark(agent, step)
            if ((self.total_epoch-1) * self.args.num_step + step) % self.args.update_freq == 0: 
                self.update(step)
                self.soft_update()
        for key in self.temp_rewards.keys(): self.temp_rewards[key] /= self.args.num_step
        self.writer.add_scalars('train-rew-by-epoch', self.temp_rewards, self.total_epoch)
        

    def test(self, render, epoch):
        self.temp_rewards = {}
        for step in range(self.args.num_step):
            self.interact(render, 'net', epoch, step, 'test')
            if self.args.test_mode:
                for agent in self.env.world.agents: self.benchmark(agent, step)
        for key in self.temp_rewards.keys(): self.temp_rewards[key] /= self.args.num_step
        self.writer.add_scalars('test-rew-by-epoch', self.temp_rewards, epoch)

    def interact(self, render, action_type, epoch, step, mode):
        image = self.render(render, step)
        image = torch.tensor(image).type(self.FT)
        self.memory.add_image(image)
        if epoch is not None and step is not None: self.args.noise *= self.args.decay
        with torch.no_grad():
            for agent in self.env.world.agents:
                name = agent.name
                obs = torch.from_numpy(self.env.observe(name)).flatten().type(self.FT)
                action = self.select_action(action_type, name, obs, image)
                self.env.step(np.array(action.cpu()))
                self.memory.add(name, 'obs', obs)
                self.memory.add(name, 'action', action)
            for name in self.env.rewards:
                if name in self.temp_rewards: self.temp_rewards[name] += self.env.rewards[name]
                else: self.temp_rewards[name] = self.env.rewards[name]
            if epoch is not None and step is not None:
                self.writer.add_scalars(f'{mode}-rew-by-step', self.env.rewards, (epoch-1) * self.args.num_step + step)
            for name in self.env.rewards:
                self.memory.add(name, 'reward', torch.tensor(self.env.rewards[name]).type(self.FT))
                obs_new = torch.from_numpy(self.env.observe(name)).flatten().type(self.FT)
                self.memory.add(name, 'obs_new', obs_new)
            self.memory.submit()# add the image,obs in to arrays

    def select_action(self, act_type, name, obs, image):
        random_action = torch.from_numpy(np.random.uniform(-1, 1, self.env.action_spaces[name].shape[0])).type(self.FT) if self.env.metadata['name']=='car_tracing' else torch.from_numpy(np.random.uniform(-1, 1, 5)).type(self.FT) 
        if act_type == 'random':
            return random_action
        if act_type == 'net':
            return self.args.noise * random_action + (1 - self.args.noise) * \
                self.actor[name](torch.unsqueeze(obs,0), torch.unsqueeze(image,0)).detach().type(self.FT).squeeze(0)

    def benchmark(self, agent, step):
        data = self.env.scenario.benchmark_data(agent, self.env.world)
        with open(f'{self.args.log_dir}/benchmark.csv', 'a') as f:
            f.write(f'{self.total_epoch},{step},{agent.name},{data}\n')

    def render(self, save, step):
        rgb_array = self.env.render()
        if save and step is not None:
            image = Image.fromarray(np.transpose(rgb_array, (1, 2, 0)))
            if not os.path.exists(f'{self.args.image_dir}/temp'): os.makedirs(f'{self.args.image_dir}/temp')
            image.save(f'{self.args.image_dir}/temp/img_{self.total_epoch}_{step}.png')
            if step+1 == self.args.num_step:
                frames = [
                    Image.open(
                        f'{self.args.image_dir}/temp/img_{self.total_epoch}_{i}.png'
                    )
                    for i in range(self.args.num_step)
                ]
                frames[0].save(f'{self.args.image_dir}/' 
                    + ('test' if self.args.test_mode else 'train') 
                    + f'_gif_{self.total_epoch}.gif', save_all=True, \
                        append_images=frames[1:], duration=1000/self.args.fps)
                shutil.rmtree(f'{self.args.image_dir}/temp')
        return rgb_array

    def sample(self):
        with torch.no_grad():
            batch, images = self.memory.sample(self.args.batch_size)
            images = torch.tensor(images).type(self.FT)
            obs_shape, cnt = 0, 0

            for agent in self.env.world.agents:
                name = agent.name
                obs_shape += self.env.observation_spaces[name].shape[0]
                cnt += self.env.action_spaces[name].shape[0]
            obs = np.ndarray((self.args.batch_size, obs_shape), dtype=np.float32)
            action = []
            obs_new = np.ndarray((self.args.batch_size, obs_shape), dtype=np.float32)
            action_new = np.ndarray((self.args.batch_size, cnt), dtype=np.float32)
            
            for i in range(self.args.batch_size):
                obs[i] = np.concatenate([batch[name]['obs'][i] for name in batch])
                action.append([torch.tensor(batch[name]['action'][i]).type(self.FT) for name in batch])
                obs_new[i] = np.concatenate([batch[name]['obs_new'][i] for name in batch])
                action_new[i] = np.concatenate([self.actor_target[name]\
                    (torch.tensor(batch[name]['obs_new'][i]).type(self.FT).unsqueeze(0), images[i].unsqueeze(0))\
                        .detach().type(self.FT).squeeze(0).cpu() for name in batch])
            obs = torch.tensor(obs).type(self.FT)
            # action = torch.tensor(action).type(self.FT)
            obs_new = torch.tensor(obs_new).type(self.FT)
            action_new = torch.tensor(action_new).type(self.FT)
            
        return batch, obs, action, obs_new, action_new, images

    def update(self,step):
        for cnt, agent in enumerate(self.env.world.agents):
            batch, obs, action, obs_new, action_new, images = self.sample()
            name = agent.name
            Q_new = self.critic_target[name](obs_new, action_new, images).detach().type(self.FT)
            Q_target = (torch.tensor(batch[name]['reward']).unsqueeze(1).type(self.FT) + self.args.gamma * Q_new).type(self.FT)
            Q = self.critic[name](obs, torch.cat([torch.cat(action[idx]) for idx in range(len(action))]).reshape((self.args.batch_size, len(torch.cat(action[0])))).type(self.FT), images).type(self.FT)
            loss_c = nn.MSELoss()(Q, Q_target).type(self.FT)
            self.writer.add_scalars("loss_critic", {name: loss_c.item()}, step + self.total_epoch * self.args.num_step)
            self.critic_optim[name].zero_grad()
            loss_c.backward()
            self.critic_optim[name].step()
            for key,value in self.critic[name].named_parameters():
                self.writer.add_scalars("grad_critic",{f"{key}_{name}": torch.mean(value.grad)}, step + self.total_epoch * self.args.num_step)
            
            temp = self.actor[name](torch.tensor(batch[name]['obs']).type(self.FT), images)
            for i in range(self.args.batch_size): action[i][cnt] = temp[i]
            loss_a = -self.critic[name](obs, torch.cat([torch.cat(action[idx]) for idx in range(len(action))]).reshape((self.args.batch_size, len(torch.cat(action[0])))).type(self.FT), images).mean().type(self.FT)
            self.writer.add_scalars("loss_actor", {name: loss_a.item()}, step + self.total_epoch * self.args.num_step)
            self.actor_optim[name].zero_grad()
            loss_a.backward()
            self.actor_optim[name].step()
            for key,value in self.actor[name].named_parameters():
                self.writer.add_scalars("grad_actor",{f"{key}_{name}": torch.mean(value.grad)}, step + self.total_epoch * self.args.num_step)

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
            self.actor[name].load_state_dict(
                torch.load(f'{self.args.model_dir}/{self.args.load_name}/actor/{name}')
            )
            self.actor_target[name].load_state_dict(self.actor[name].state_dict())
            self.critic[name].load_state_dict(
                torch.load(f'{self.args.model_dir}/{self.args.load_name}/critic/{name}')
            )
            self.critic_target[name].load_state_dict(self.critic[name].state_dict())

    def save(self):
        if not os.path.exists(f'{self.args.model_dir}/{self.total_epoch}/actor'): 
            os.makedirs(f'{self.args.model_dir}/{self.total_epoch}/actor')
        if not os.path.exists(f'{self.args.model_dir}/{self.total_epoch}/critic'): 
            os.makedirs(f'{self.args.model_dir}/{self.total_epoch}/critic')
        for name in self.actor:
            torch.save(
                self.actor[name].state_dict(),
                f'{self.args.model_dir}/{self.total_epoch}/actor/{name}',
            )
            torch.save(
                self.critic[name].state_dict(),
                f'{self.args.model_dir}/{self.total_epoch}/critic/{name}',
            )