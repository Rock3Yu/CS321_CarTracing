import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, env, name):
        super().__init__()
        obs_shape = env.observation_spaces[name].shape[0]
        act_shape = env.action_spaces[name].shape[0]
        self.fc1 = nn.Linear(obs_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, act_shape)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, obs): 
        obs = self.fc1(obs)
        obs = self.relu(obs)
        obs = self.fc2(obs)
        obs = self.relu(obs)
        obs = self.fc3(obs)
        return self.softmax(obs)

class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        input_shape = 0
        for agent in env.world.agents:
            name = agent.name
            obs_shape = env.observation_spaces[name].shape[0]
            act_shape = env.action_spaces[name].shape[0]
            input_shape += obs_shape + act_shape
        self.fc1 = nn.Linear(input_shape,128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        
    def forward(self, obs, actions):
        res = torch.cat([obs, actions], 1)
        res = self.fc1(res)
        res = self.relu(res)
        res = self.fc2(res)
        res = self.relu(res)
        return self.fc3(res)
