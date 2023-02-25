import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import Astar_Policy as astar_p

import Car_Tracing_Scenario

class MADDPG_Policy:
    def __init__(self, mode,num_police, num_criminal,num_escape,gpu=False):#mode = Train,Test
        self.gpu=gpu
        self.mode=0 if mode=='Train' else 1  #'Test'=0 or 'Train'=1
        self.num_police = num_police
        self.num_criminal = num_criminal
        self.num_agent = num_police + num_criminal
        self.num_escape=num_escape
        self.done=[False]*self.num_agent
        self.is_criminal=[False] * self.num_agent
        self.actors = [None] * self.num_agent #actor neural
        self.actors_target = [None] * self.num_agent #actor target neurual
        self.critics = [None] * self.num_agent #critic neural
        self.critics_target = [None] * self.num_agent
        self.optimizer_critics = [None] * self.num_agent
        self.optimizer_actors = [None] * self.num_agent
        self.memory = self.Memory(1000)
        self.tao = .01 #学习率
        self.gamma = .97 #贴现因子，考虑越长远，给现在的影响越小

        for i in range(self.num_agent):
            self.actors[i] = self.Actor()
            self.actors_target[i] = self.Actor()
            self.critics[i] = self.Critic()
            self.critics_target[i] = self.Critic()
            self.optimizer_actors[i] = optim.Adam(self.actors[i].parameters(), lr=1.) # 学习率1.直接更新，momentum动量
            self.optimizer_critics[i] = optim.Adam(self.critics[i].parameters(), lr=1.)
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

    def act(self,num_epoch, cycles, train_batch): 
        if self.gpu:
            self.importcuda()
        #env = Car_Tracing_Scenario.env(map_path = 'map/test_map_d.txt', max_cycles = cycles, ratio = .9)#4
        #env = Car_Tracing_Scenario.env(map_path = 'map/test_map.txt', max_cycles = cycles, ratio = 1.5)#4
        env = Car_Tracing_Scenario.env(map_path = 'map/city.txt', max_cycles = cycles, ratio = .9)#4
        # self.load()
        for epoch in range(num_epoch): #the epoch is a training process 1个epoch等于使用训练集中的全部样本训练一次，通俗的讲epoch的值就是整个数据集被轮几次。
            env.reset()
            self.done=[False]*self.num_agent
            obs = (env.last()[0]['obs'])
            print('epoch',epoch)
            for step in range(cycles): #the step is a step in max-steps
                # print(f'step:{step}:',self.done)
                if (step+1) % 50 == 0:
                    print(f'training...  step={step+1}/{cycles}' if self.mode==0 else f'testing...  step={step+1}/{cycles}')
                actions = []
                rewards = []
                done_cnt=0

                # env.render()
                for i in range(self.num_agent):  
                    if 'criminal' in env.last()[0]['name']:
                        self.is_criminal[i]=True
                        
                        # print(env.rewards[env.last()[0]['rew']])
                        if abs(env.rewards[env.last()[0]['rew']])==100 or self.done[i]:
                            self.done[i]=True
                            action=np.array([0.,0.])
                            done_cnt+=1
                        else:
                            action=self.ToAstar(obs)
                    else: 
                        if  self.gpu:
                            action = self.actors[i](torch.as_tensor(obs,device='cuda:0',dtype=torch.float)).detach().cpu().numpy() 
                        else:
                            action = self.actors[i](torch.as_tensor(obs,dtype=torch.float)).detach().numpy()
                            # 即返回一个新的tensor，从当前计算图中分离下来。 但是仍指向原变量 的 存放位置，不能用于计算梯度
                    action = action/np.max(np.abs(action)) if np.max(np.abs(action))!=0 else np.array([0.,0.])  #normalize
                    env.step(action)
                    if self.mode==0: #Train#################################
                        actions.append(action)
                        rewards.append(env.rewards[env.last()[0]['rew']])
                if done_cnt== self.num_criminal: #if all criminal is catched
                    print(done_cnt,self.num_criminal)
                    print(f'epoch {epoch} ends at step {step+1}')
                    break
                if self.mode==1:#Test
                    obs= np.array(env.last()[0]['obs'])
                if self.mode==0:#Train#########################################
                    obs_new = env.last()[0]['obs']
                    if 100 in rewards or -100 in rewards:
                        print(rewards)
                    self.memory.add(obs, actions, rewards, obs_new)  #the buffer for all agents
                    obs = obs_new
                    if step > 3:
                        for cnt in range(train_batch):#训练的batch是数目，每次选择多少条sample训练
                            for i in range(1,self.num_agent):
                                if self.is_criminal[i]:
                                    continue
                                obs, actions, rewards, obs_new = self.memory.sample() #随机选择buffer中的一组数据
                                actions_new = []
                                for j in range(self.num_agent):
                                    if  self.gpu:
                                        action_new = self.actors_target[i](torch.as_tensor(obs_new,device='cuda:0',dtype=torch.float)).detach().cpu().numpy() 
                                    else:
                                        action_new = self.actors_target[j](torch.as_tensor(obs_new,dtype=torch.float)).detach().numpy() #对每个智体确定他们的下一步动作,根据A_target,μ'
                                    action_new /= np.max(np.abs(action_new)) #将动作normalize归一化
                                    actions_new.append(action_new)
                                if self.gpu:
                                    
                                    Q = self.critics[i](torch.as_tensor(obs,device='cuda:0',dtype=torch.float)
                                        , torch.as_tensor(actions,device='cuda:0',dtype=torch.float))  #找到非冻结的Q函数根据obs和actions
                                    Q_target = self.critics_target[i](torch.as_tensor(obs_new,device='cuda:0',dtype=torch.float)
                                        , torch.as_tensor(actions_new,device='cuda:0',dtype=torch.float)) #μ'目标Q函数，即冻结的Q函数
                                    y = torch.tensor(rewards[i],device='cuda:0') + self.gamma * Q_target #求出'真实'的Qvalue=y
                                    self.optimizer_critics[i].zero_grad() # 将Critic/Q的参数梯度初始化为0
                                    loss_critic = nn.MSELoss()(Q, y) #不缩减维度，计算MSE平方误差，（相减求平方）返回的是tensor
                                    loss_critic.backward()  # 反向传播计算梯度,将损失loss 向输入侧进行反向传播
                                    self.optimizer_critics[i].step()  # 将Q函数的param进行优化更新，要沿着梯度的反方向调整变量值以减少Cost(通过Adam参数优化器)
                                    self.optimizer_actors[i].zero_grad()  #将Actor模型的参数梯度初始化为0
                                    loss_actor = -torch.mean(self.critics[i](torch.as_tensor(obs,device='cuda:0',dtype=torch.float)
                                        , torch.as_tensor(actions,device='cuda:0',dtype=torch.float))) #求critic网络的均值，返回的是标量
                                    loss_actor.backward() # 反向传播计算梯度
                                    self.optimizer_actors[i].step() #将μ函数（策略函数）的参数进行更新，更新μ
                                else:
                                    Q = self.critics[i](torch.as_tensor(obs,dtype=torch.float)
                                        , torch.as_tensor(actions,dtype=torch.float))  #找到非冻结的Q函数根据obs和actions
                                    Q_target = self.critics_target[i](torch.as_tensor(obs_new,dtype=torch.float)
                                        , torch.as_tensor(actions_new,dtype=torch.float)) #μ'目标Q函数，即冻结的Q函数
                                    y = torch.tensor(rewards[i]) + self.gamma * Q_target #求出'真实'的Qvalue=y
                                    self.optimizer_critics[i].zero_grad() # 将Critic/Q的参数梯度初始化为0
                                    loss_critic = nn.MSELoss()(Q, y) #不缩减维度，计算MSE平方误差，（相减求平方）返回的是tensor
                                    loss_critic.backward()  # 反向传播计算梯度,将损失loss 向输入侧进行反向传播
                                    self.optimizer_critics[i].step()  # 将Q函数的param进行优化更新，要沿着梯度的反方向调整变量值以减少Cost(通过Adam参数优化器)
                                    self.optimizer_actors[i].zero_grad()  #将Actor模型的参数梯度初始化为0,防止爆显存
                                    loss_actor = -torch.mean(self.critics[i](torch.as_tensor(obs,dtype=torch.float)
                                        , torch.as_tensor(actions,dtype=torch.float))) #求critic网络的均值，返回的是标量
                                    loss_actor.backward() # 反向传播计算梯度
                                    self.optimizer_actors[i].step() #将μ函数（策略函数）的参数进行更新，更新μ
                        for i in range(1,self.num_agent):  #在执行完batchsize的训练后，通过τ学习率更新Q'的参数
                            cur = self.critics[i].state_dict()
                            tar = self.critics_target[i].state_dict()
                            for key in cur.keys():
                                tar[key] = self.tao * cur[key] + (1 - self.tao) * tar[key] #更新Q’
                            self.critics_target[i].load_state_dict(tar)
                            cur = self.actors[i].state_dict()
                            tar = self.actors_target[i].state_dict()
                            for key in cur.keys():
                                tar[key] = self.tao * cur[key] + (1 - self.tao) * tar[key] #更新μ'
                            self.actors_target[i].load_state_dict(tar)
            if self.mode==0:
                self.save()
    
    def ToAstar(self,obs):
        # start=time.time()
        isPolicy=[True]*self.num_police+[False]*self.num_criminal
        idx=self.num_police  #which is criminal
        p_pos=((obs[self.num_police][0]),(obs[self.num_police][1]))#criminal position
        criminal_state=True
        police_pos=[]
        criminal_pos=[]
        landmark_pos=[]
        escape_pos=[]
        map_height=200
        map_width=250
        map=np.full((map_height,map_width),"",dtype=str)
        for index,i in enumerate(obs) :
            if index< self.num_police:
                if i[0]<0 or i[0]>=map_height or i[1]<0 or i[1]>=map_width:
                    return np.array([0.,0.])
                map[int(i[0]),int(i[1])]='p'
                police_pos.append(((i[0]),(i[1])))
            elif index<self.num_agent:
                if i[0]<0 or i[0]>=map_height or i[1]<0 or i[1]>=map_width:
                    return np.array([0.,0.])
                map[int(i[0]),int(i[1])]='c'
                criminal_pos.append(((i[0]),(i[1])))
            elif index>=len(obs)-self.num_escape:
                map[int(i[0]),int(i[1])]='e'
                escape_pos.append(((i[0]),(i[1])))
            else:   #landmark
                map[int(i[0]),int(i[1])]='o'
        result = []
        result.append(isPolicy)
        result.append(idx)
        result.append(np.array(p_pos))
        result.append(criminal_state)
        result.append(np.array(police_pos))
        result.append(np.array(criminal_pos))
        result.append(landmark_pos)
        result.append(np.array(escape_pos))
        result.append(map)
        # mid=time.time()
        a_policy=astar_p.astarPolicy(result)
        action = a_policy.act()
        # end=time.time()
        # print('cost at prepare:',mid-start,', cost at astar:',end-mid)
        return action

    def save(self):
        for i in range(self.num_agent):
            print(f'saving...  idx={i}')
            torch.save(self.actors[i].state_dict(), f'./data/actors/{i}')
            torch.save(self.actors_target[i].state_dict(), f'./data/actors_target/{i}')
            torch.save(self.critics[i].state_dict(), f'./data/critics/{i}')
            torch.save(self.critics_target[i].state_dict(), f'./data/critics_target/{i}')

    def load(self):
        for i in range(self.num_agent):
            print(f'loading...  idx={i}')
            self.actors[i].load_state_dict(torch.load(f'./data/actors/{i}'))
            self.actors_target[i].load_state_dict(torch.load(f'./data/actors_target/{i}'))
            self.critics[i].load_state_dict(torch.load(f'./data/critics/{i}'))
            self.critics_target[i].load_state_dict(torch.load(f'./data/critics_target/{i}'))

            #input 3428*2 dim, output 2 dim stands for x,y direction (velocity is judg e by max_speed)

    def importcuda(self):
        print('CUDA: ',torch.cuda.is_available())
        for i in range(self.num_agent):
            self.actors[i]=self.actors[i].cuda()
            self.actors_target[i]=self.actors_target[i].cuda()
            self.critics[i]=self.critics[i].cuda()
            self.critics_target[i]=self.critics_target[i].cuda()
        # self.cuda()
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3478 * 2, 3478 * 2)
            self.fc2 = nn.Linear(3478 * 2, 3478)
            self.fc3 = nn.Linear(3478, 2)
            self.relu = nn.ReLU()

        def forward(self, obs):   #############问题在这里
            # obs = torch.tensor(np.array(obs.cpu()), dtype=torch.float,device='cuda:0').flatten()
            obs = obs.flatten()   
            #将latten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
            obs = self.fc1(obs)
            obs = self.relu(obs)
            obs = self.fc2(obs)
            obs = self.relu(obs)
            return self.fc3(obs)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1_1 = nn.Linear(3478 * 2, 3478 * 2)
            self.fc1_2 = nn.Linear(3478 * 2, 3478)
            self.fc2_1 = nn.Linear(4 * 2, 16)
            self.fc2_2 = nn.Linear(16, 8)
            self.fc3_1 = nn.Linear(3486, 3486 * 2)
            self.fc3_2 = nn.Linear(3486 * 2, 3486)
            self.fc3_3 = nn.Linear(3486, 1)
            self.relu = nn.ReLU()

        def forward(self, obs,action):
            # obs = torch.tensor(np.array(obs), dtype=torch.float).flatten()
            # action = torch.tensor(np.array(action), dtype=torch.float).flatten()
            obs=obs.flatten()
            action=action.flatten()
            obs = self.fc1_1(obs)
            obs = self.relu(obs)
            obs = self.fc1_2(obs)
            obs = self.relu(obs)
            action = self.fc2_1(action)
            action = self.relu(action)
            action = self.fc2_2(action)
            action = self.relu(action)
            res = torch.cat([obs, action])
            res = self.fc3_1(res)
            res = self.relu(res)
            res = self.fc3_2(res)
            res = self.relu(res)
            return self.fc3_3(res)

    class Memory:
        def __init__(self, size):
            self.size = size
            self.buffer_obs = [None] * size
            self.buffer_actions = [None] * size
            self.buffer_rewards = [None] * size
            self.buffer_obs_new = [None] * size
            self.cur_idx = 0
            self.total_cnt = 0

        def add(self, obs, actions, rewards, obs_new):
            self.buffer_obs[self.cur_idx] = obs
            self.buffer_actions[self.cur_idx] = actions
            self.buffer_rewards[self.cur_idx] = rewards
            self.buffer_obs_new[self.cur_idx] = obs_new
            self.cur_idx = (self.cur_idx + 1) % self.size
            self.total_cnt += 1

        def sample(self): #随机选择一个buffer中的数据
            if self.total_cnt < self.size:
                idx = random.randrange(0, self.total_cnt)
            else:
                idx = random.randrange(0, self.size)
            return self.buffer_obs[idx], self.buffer_actions[idx], self.buffer_rewards[idx], self.buffer_obs_new[idx]
        
# def parse_args():
#     parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
#     # Environment
#     parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
#     parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
#     parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
#     parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
#     parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
#     parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
#     # Core training parameters
#     parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
#     parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
#     parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
#     parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
#     # Checkpointing
#     parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
#     parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
#     parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
#     parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
#     # Evaluation
#     parser.add_argument("--restore", action="store_true", default=False)
#     parser.add_argument("--display", action="store_true", default=False)
#     parser.add_argument("--benchmark", action="store_true", default=False)
#     parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
#     parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
#     parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
#     return parser.parse_args()

if __name__ == '__main__':
    policy = MADDPG_Policy('Train',3, 1,1,gpu=False)
    # policy = MADDPG_Policy('Test',3, 1,1,gpu=False)
    policy.act(100, 250, 5)
