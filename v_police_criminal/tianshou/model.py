import torch
from torch import nn
from tianshou.data.batch import Batch
import numpy as np
# cuda0 = torch.device('cuda:0')
class Actor(nn.Module):
    def __init__(self, env, name):
        super().__init__()
        obs_shape = env.observation_spaces[name].shape[0]
        act_shape = env.action_spaces[name].shape[0]
        self.fc_image = nn.Linear(9*9*64, 256)
        self.fc_obs = nn.Linear(obs_shape, 256)
        # self.fc1 = nn.Linear(obs_shape + 9*9*64, 512)
        self.fc1 = nn.Linear( 256, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, act_shape)
        self.conv1 = nn.Conv2d(1, 32, 4, stride=3)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 0)
        self.sigmoid = nn.Sigmoid()
        self.device='cpu'
        if torch.cuda.is_available():
            self.device='cuda:0'
            self.to('cuda:0')


    def forward(self, obs,state=None,info={}):
        if isinstance(obs,Batch):
            obs=obs['obs']
        image = [ob[1] for ob in obs];image=np.array(image,dtype=float)
        obs= [ob[0] for ob in obs];obs=np.array(obs,dtype=float)
        obs = torch.as_tensor(obs,device=self.device,dtype=torch.float32)
        image= torch.as_tensor(image,device=self.device,dtype=torch.float32)
        if len(obs.shape)==1:
            obs=torch.unsqueeze(obs,0);image=torch.unsqueeze(obs,0)
        image = self.conv1(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.relu(image)
        image = torch.flatten(image, 1,-1)
        image = self.fc_image(image)
        # obs = self.fc_obs(obs)
        # image.fill_(0) 
        # obs.fill_(0)
        # obs = torch.concat((obs, image), 1)
        obs = self.fc1(image)
        obs = self.relu(obs)
        obs = self.fc2(obs)
        obs = self.relu(obs)
        obs = self.fc3(obs)
        return self.sigmoid(obs),state

class Critic(nn.Module):
    def __init__(self, env,name):
        super().__init__()
        input_shape = 0
        for agent in env.world.agents:
            name = agent.name
            obs_shape = env.observation_spaces[name].shape[0]
            act_shape = env.action_spaces[name].shape[0]
            input_shape += obs_shape + act_shape
        self.fc_image = nn.Linear(9*9*64, 256)
        self.fc_obs = nn.Linear(input_shape, 256)
        self.fc1 = nn.Linear( 256 , 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.conv1 = nn.Conv2d(1, 32, 4, stride=3)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.relu = nn.ReLU()
        self.device='cpu'
        if torch.cuda.is_available():
            self.device='cuda:0'
            self.to('cuda:0')

    def forward(self, obs, actions,state=None,info={}):
        if isinstance(obs,Batch):
            obs=obs['obs']
        image = [ob[1] for ob in obs];image=np.array(image,dtype=float)
        obs= [ob[0] for ob in obs];obs=np.array(obs,dtype=float)
        obs = torch.as_tensor(obs,device=self.device,dtype=torch.float32,)
        image= torch.as_tensor(image,device=self.device,dtype=torch.float32,)
        if len(obs.shape)==1:
            obs=torch.unsqueeze(obs,0);image=torch.unsqueeze(obs,0)
        actions=torch.as_tensor(actions,device=self.device)
        image = self.conv1(image)
        image = self.relu(image)
        image = self.conv2(image)
        image = self.relu(image)
        image = torch.flatten(image, 1, -1)
        image = self.fc_image(image)
        # image.fill_(0)
        # res = torch.cat([obs, actions], 1)
        # res = self.fc_obs(res)
        # res.fill_(0)
        # res = torch.cat((res, image), 1)
        res = self.fc1(image)
        res = self.relu(res)
        res = self.fc2(res)
        res = self.relu(res)
        return self.fc3(res)  #need to be same as nn.Net
