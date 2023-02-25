import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pettingzoo.mpe.simple.simple as simple
from torch.utils.tensorboard import SummaryWriter
import os

# ！！！请务必在每次实验前更改exp_1名字，以记录实验于tensorboard中
TENSORBOARD_PATH = './logs/simple_10000_batch10_interval5/'

# dim=4+(n-1)*2+entity*2
#agent.adversary = True if i < num_adversaries else False
class MADDPG_Policy:
    def __init__(self, mode,num_police, num_criminal,num_escape,gpu=False):  # mode = Train,Test
        if not os.path.exists(TENSORBOARD_PATH):  # 如果路径不存在
            os.makedirs(TENSORBOARD_PATH)
        self.writer = SummaryWriter(TENSORBOARD_PATH)
        self.gpu=gpu
        self.mode=0 if mode=='Train' else 1  #'Test'=0 or 'Train'=1
        self.num_police = num_police
        self.num_criminal = num_criminal
        self.num_agent = num_police + num_criminal
        self.num_escape=num_escape
        self.num_entity=1
        self.done=[False]*self.num_agent
        self.is_criminal=[False] * self.num_agent
        self.actors = [None] * self.num_agent  # actor neural
        self.actors_target = [None] * self.num_agent # actor target neurual
        self.critics = [None] * self.num_agent  # critic neural
        self.critics_target = [None] * self.num_agent
        self.optimizer_critics = [None] * self.num_agent
        self.optimizer_actors = [None] * self.num_agent
        self.memory = self.Memory(100000, 2+2*self.num_entity, 5 ,self.num_agent)
        self.actionMemory=[[0., 0.] * 20] * self.num_agent
        self.tao = .01  # 学习率
        self.gamma = .95  # 贴现因子，考虑越长远，给现在的影响越小
        self.reward_record=[]
        self.touches=None
        for i in range(self.num_agent):
            self.actors[i] = self.Actor(num_othernotadv=num_police-1,num_agent=self.num_agent,num_entity=self.num_entity)
            self.actors_target[i] = self.Actor(num_othernotadv=num_police-1,num_agent=self.num_agent,num_entity=self.num_entity)
            self.critics[i] = self.Critic(num_othernotadv=num_police-1,num_agent=self.num_agent,num_entity=self.num_entity)
            self.critics_target[i] = self.Critic(num_othernotadv=num_police-1,num_agent=self.num_agent,num_entity=self.num_entity)
            self.optimizer_actors[i] = optim.Adam(self.actors[i].parameters(), lr=1.)  # 学习率1.直接更新，momentum动量
            self.optimizer_critics[i] = optim.Adam(self.critics[i].parameters(), lr=1.)
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

    def act(self,num_epoch, cycles, train_batch,update_interval,ifRender, load):
        if self.gpu:
            self.importcuda()
        # env = Car_Tracing_Scenario.env(map_path = 'map/test_map_d.txt', max_cycles = cycles, ratio = .9)  # 4
        # env = Car_Tracing_Scenario.env(map_path = 'map/test_map.txt', max_cycles = cycles, ratio = 1.5)  # 4
        env = simple.env(max_cycles=cycles*self.num_agent,continuous_actions= True)
        for i in range(num_epoch):
            self.reward_record.append([]) #加入每一轮的结果
        if load:
            self.load()
        self.touches=np.zeros(num_epoch)
        def action_to_pos(actions):
            if len(actions.shape)>1:
                for action in actions:
                    action[0]=0
                    action[1]=action[1]-action[2];action[2]=0
                    action[3]=action[3]-action[4];action[4]=0
                    if action[1]<0:
                        action[2]=-action[1]
                        action[1]=0
                    if action[3]<0:
                        action[4]=-action[3]
                        action[3]=0
                return actions
            else :
                action=actions
                action[0]=0
                action[1]=action[1]-action[2];action[2]=0
                action[3]=action[3]-action[4];action[4]=0
                if action[1]<0:
                    action[2]=-action[1]
                    action[1]=0
                if action[3]<0:
                    action[4]=-action[3]
                    action[3]=0
                return action
        rate=0 #the successfully rate of reaching entity
        for epoch in range(num_epoch):  # the epoch is a training process 1个epoch等于使用训练集中的全部样本训练一次，通俗的讲epoch的值就是整个数据集被轮几次。
            env.reset()
            min_dis=1000

            self.done=[False]*self.num_agent
            print('epoch',epoch)
            start=time.time()
            for step in range(cycles):  # the step is a step in max-steps
                # start=time.time()
                action_idx = step % 20
                if (step+1) %100 == 0:
                    print(f'training simple...  step={step+1}/{cycles}' if self.mode==0 else f'testing...  step={step+1}/{cycles}')
                if ifRender:
                    env.render()
                rewards=[];obs_agents=[];actions = [];obs_new_agents=[]
                for i in range(self.num_agent):
                    # TODO:计算真正的obs，计算视野范围，加动作序列

                    obs=env.last()[0]
                    obs=np.asarray(obs,dtype=np.float16)
                    if  self.gpu:
                        action = self.actors[i](torch.as_tensor( obs,device='cuda:0',dtype=torch.float16)).detach().cpu().numpy()
                    else:
                        action = self.actors[i](torch.as_tensor(obs,dtype=torch.float16)).detach().numpy()
                            # 即返回一个新的tensor，从当前计算图中分离下来。 但是仍指向原变量 的 存放位置，不能用于计算梯度
                    action=action_to_pos(action+np.random.normal(0,1,5)*0.5)
                    action = action/np.max(np.abs(action)) if np.max(np.abs(action))!=0. else np.array([0.,0.,0.,0.,0.])  #normalize
                    action= np.nan_to_num(action)
                    self.actionMemory[i][action_idx] = action  #更新记忆动作序列
                    env.step(action)
                    # col[epoch]+=env.benchmark(env.agent_selection)
                    if env.agent_selection[0:3]=="adv" and type(env.last()[3])==int:
                        self.touches[epoch]+=env.last()[3]
                    #0left,right,up,down
                    if self.mode==0:  # Train#################################
                        actions.append(action)
                        obs_agents.append(obs)
                        if i==self.num_agent-1:
                            for j in range(self.num_agent):
                                rewards.append(env.last()[1])
                                # print(env.last()[1],env.last()[0])
                                obs_new_agents.append(env.last()[0])
                                env.agent_selection=env.agents[(j+1)%self.num_agent]

                action_idx = (action_idx+1)%20  # update action idx
                ###################
                min_dis=min(-min(rewards),min_dis)
                self.reward_record[epoch].append(rewards)
                # print("step,cost",time.time()-start);start=time.time()
                ###################
                # if self.mode==1:#Test
                #     obs= np.array(env.last()[0]['obs'])
                if self.mode==0:  # Train#########################################
                    #TODO:计算真正的obs_new，计算视野范围，加动作序列
                    actions=np.asarray(actions);rewards=np.asarray(rewards);obs_agents=np.asarray(obs_agents);obs_new_agents=np.asarray(obs_new_agents)
                    self.memory.add(obs_agents, actions, rewards, obs_new_agents)  #the buffer for all agents

                    if step > 19 and step % update_interval ==0 :
                        for cnt in range(1):#训练的batch是数目，每次选择多少条sample训练
                            for i in range(self.num_agent):
                                obs_agents, actions, rewards, obs_new_agents = self.memory.sample(train_batch) #随机选择buffer中的一组数据
                                obs=obs_agents[:,i];obs_new=obs_new_agents[:,i]

                                actions_new = []
                                for j in range(self.num_agent):

                                    if  self.gpu:
                                        action_new = self.actors_target[i](torch.as_tensor(obs_new,device='cuda:0',dtype=torch.float16)).detach().cpu().numpy()
                                    else:
                                        action_new = self.actors_target[j](torch.as_tensor((obs_new),dtype=torch.float16)).detach().numpy() #对每个智体确定他们的下一步动作,根据A_target,μ'
                                    action_new=action_to_pos(action_new)
                                    action_new = action_new/np.max(np.abs(action_new)) if np.max(np.abs(action))!=0. else np.array([0.,0.,0.,0.,0.]) #将动作normalize归一化
                                    # if not self.is_criminal[i]:
                                    #     action_new=np.nan_to_num(action_new)
                                    #     action_new = action_new/np.max(np.abs(action_new)) if np.max(np.abs(action_new))!=0. else np.array([0.,0.,0.,0.,0.])  #normalize
                                    # else:
                                    action_new=np.nan_to_num(action_new)

                                    # action_new=np.nan_to_num(action)+np.random.normal(0, 1, 5)  #此时不需要加入噪声
                                    actions_new.append(action_new)
                                actions_new=np.transpose(np.array(actions_new,dtype=np.float16),(1,0,2))
                                if self.gpu:

                                    Q = self.critics[i](torch.as_tensor(obs_agents,device='cuda:0',dtype=torch.float16)
                                        , torch.as_tensor(actions,device='cuda:0',dtype=torch.float16))  #找到非冻结的Q函数根据obs和actions
                                    Q_target = self.critics_target[i](torch.as_tensor(obs_new_agents,device='cuda:0',dtype=torch.float16)
                                        , torch.as_tensor(actions_new,device='cuda:0',dtype=torch.float16)) #μ'目标Q函数，即冻结的Q函数
                                    y = torch.tensor(rewards[i],device='cuda:0') + self.gamma * Q_target #求出'真实'的Qvalue=y
                                    self.optimizer_critics[i].zero_grad() # 将Critic/Q的参数梯度初始化为0
                                    loss_critic = nn.MSELoss()(Q, y) #不缩减维度，计算MSE平方误差，（相减求平方）返回的是tensor
                                    loss_critic=torch.mean(loss_critic)

                                    loss_critic.backward()  # 反向传播计算梯度,将损失loss 向输入侧进行反向传播
                                    self.optimizer_critics[i].step()  # 将Q函数的param进行优化更新，要沿着梯度的反方向调整变量值以减少Cost(通过Adam参数优化器)
                                    self.optimizer_actors[i].zero_grad()  #将Actor模型的参数梯度初始化为0
                                    loss_actor = -torch.mean(self.critics[i](torch.as_tensor(obs_agents,device='cuda:0',dtype=torch.float16)
                                        , torch.as_tensor(actions,device='cuda:0',dtype=torch.float16))) #求critic网络的均值，返回的是标量
                                    loss_actor.backward() # 反向传播计算梯度
                                    self.optimizer_actors[i].step() #将μ函数（策略函数）的参数进行更新，更新μ
                                else:
                                    Q = self.critics[i](torch.as_tensor(obs_agents,dtype=torch.float16)
                                        , torch.as_tensor(actions,dtype=torch.float16))  #找到非冻结的Q函数根据obs和actions
                                    Q_target = self.critics_target[i](torch.as_tensor(obs_new_agents,dtype=torch.float16)
                                        , torch.as_tensor(actions_new,dtype=torch.float16)) #μ'目标Q函数，即冻结的Q函数
                                    y = torch.tensor(rewards[i]) + self.gamma * Q_target #求出'真实'的Qvalue=y
                                    self.optimizer_critics[i].zero_grad() # 将Critic/Q的参数梯度初始化为0
                                    loss_critic = nn.MSELoss()(Q, y) #不缩减维度，计算MSE平方误差，（相减求平方）返回的是tensor
                                    loss_critic=torch.mean(loss_critic)
                                    loss_critic.backward()  # 反向传播计算梯度,将损失loss 向输入侧进行反向传播
                                    self.optimizer_critics[i].step()  # 将Q函数的param进行优化更新，要沿着梯度的反方向调整变量值以减少Cost(通过Adam参数优化器)
                                    self.optimizer_actors[i].zero_grad()  #将Actor模型的参数梯度初始化为0,防止爆显存
                                    loss_actor = -torch.mean(self.critics[i](torch.as_tensor(obs_agents,dtype=torch.float16)
                                        , torch.as_tensor(actions,dtype=torch.float16))) #求critic网络的均值，返回的是标量
                                    loss_actor.backward() # 反向传播计算梯度
                                    self.optimizer_actors[i].step() #将μ函数（策略函数）的参数进行更新，更新μ
                        # print("network update",time.time()-start);start=time.time()
                        for i in range(0,self.num_agent-1):  #在执行完batchsize的训练后，通过τ学习率更新Q'的参数
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
                        # print("parameter update",time.time()-start);start=time.time()
            if self.mode==0:
                self.save()
                print("save cost",time.time()-start)
            if min_dis<1:
                rate+=1
            # save to tensorboard
            tag0 = 'police0'
            tag1 = 'police1'
            tag2 = 'police2'
            tag_r ="rate_reach"
            # tag3 = 'critical'
            scalar_value0 = self.reward_record[epoch][-1][0]
            scalar_value1 = self.reward_record[epoch][-1][1]
            scalar_value2 = self.reward_record[epoch][-1][2]
            # scalar_value3 = self.reward_record[epoch][-1][3]
            self.writer.add_scalar(tag0, scalar_value0, epoch)
            self.writer.add_scalar(tag1, scalar_value1, epoch)
            self.writer.add_scalar(tag2, scalar_value2, epoch)
            # self.writer.add_scalar(tag2, rate/num_epoch, epoch)
            # self.writer.add_scalar(tag3, scalar_value3, epoch)

    def to1D(self,list2):
        list1=[]
        for t in list2:
            list1.append(t[0])
            list1.append(t[1])
        return list1

    def save(self):
        for i in range(self.num_agent):
            print(f'saving...  idx={i}')
            torch.save(self.actors[i].state_dict(), f'./data_simple/actors/{i}')
            torch.save(self.actors_target[i].state_dict(), f'./data_simple/actors_target/{i}')
            torch.save(self.critics[i].state_dict(), f'./data_simple/critics/{i}')
            torch.save(self.critics_target[i].state_dict(), f'./data_simple/critics_target/{i}')

    def load(self):
        path='./data_simple/'
        for i in range(self.num_agent):
            print(f'loading...  idx={i}')
            if self.gpu:
                self.actors[i].load_state_dict(torch.load(path+f'actors/{i}'))
                self.actors_target[i].load_state_dict(torch.load(path+f'actors_target/{i}'))
                self.critics[i].load_state_dict(torch.load(path+f'critics/{i}'))
                self.critics_target[i].load_state_dict(torch.load(path+f'critics_target/{i}'))
            else:
                self.actors[i].load_state_dict(torch.load(path+f'actors/{i}','cpu'))
                self.actors_target[i].load_state_dict(torch.load(path+f'actors_target/{i}','cpu'))
                self.critics[i].load_state_dict(torch.load(path+f'critics/{i}','cpu'))
                self.critics_target[i].load_state_dict(torch.load(path+f'/critics_target/{i}','cpu'))

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
        def __init__(self,num_othernotadv,num_agent,num_entity):
            super().__init__()
            input1=2+num_entity*2
            self.fc1 = nn.Linear( input1, 128 )
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 5)
            self.relu = nn.ReLU()

        def forward(self, obs):   #############问题在这里
            # obs = torch.tensor(np.array(obs.cpu()), dtype=torch.float16,device='cuda:0').flatten()
            if len(obs.shape)>1:
                obs = obs.flatten(1)
            #将latten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
            obs = self.fc1(obs)
            obs = self.relu(obs)
            obs = self.fc2(obs)
            obs = self.relu(obs)
            return self.fc3(obs)

    class Critic(nn.Module):
        def __init__(self,num_othernotadv,num_agent,num_entity):
            super().__init__()
            input1=(2+num_entity*2)*num_agent+num_agent*5
            self.fc3_1 = nn.Linear(input1,128)
            self.fc3_2 = nn.Linear(128, 128)
            self.fc3_3 = nn.Linear(128, 1)
            self.relu = nn.ReLU()

        def forward(self, obs:torch.Tensor,action:torch.Tensor):

            obs=obs.flatten(1);action=action.flatten(1)
            # print(obs.shape,action.shape)
            res = torch.cat([obs, action],1)
            # print(res.shape)
            res = self.fc3_1(res)
            res = self.relu(res)
            res = self.fc3_2(res)
            res = self.relu(res)
            return self.fc3_3(res)

    class Memory:
        def __init__(self, size,obs_dim,action_dim,agent_num):
            self.size = size
            self.buffer_obs =np.zeros((size,agent_num,obs_dim),dtype=np.float16)
            self.buffer_actions = np.zeros((size,agent_num,action_dim),dtype=np.float16)
            self.buffer_rewards = np.zeros((size,agent_num),dtype=np.float16)
            self.buffer_obs_new = np.zeros((size,agent_num,obs_dim),dtype=np.float16)
            self.cur_idx = 0
            self.total_cnt = 0

        def add(self, obs, actions, rewards, obs_new):
            self.buffer_obs[self.cur_idx] = obs
            self.buffer_actions[self.cur_idx] = actions
            self.buffer_rewards[self.cur_idx] = rewards
            self.buffer_obs_new[self.cur_idx] = obs_new
            self.cur_idx = (self.cur_idx + 1) % self.size
            self.total_cnt += 1

        def sample(self,train_batch): #随机选择一个buffer中的数据
            if self.total_cnt < self.size:
                idx = np.random.choice(self.total_cnt,size=train_batch,replace=False)
            else:
                idx=np.random.choice(self.size, size=train_batch, replace=False)

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


def printReward(policy):
    print_log = open("/home/maddpg/v1/MPE/reward_s.txt",'w')
    print(str(policy.reward_record),file = print_log)
    print_log.close()  # output reward to file，array of len num_epoch

def printTouch(policy):
    print_log = open("/home/maddpg/v1/MPE/touches_s.txt",'w')
    print(str(policy.touches),file = print_log)
    print_log.close()  # output reward to file，array of len num_epoch

if __name__ == '__main__':
    # policy = MADDPG_Policy('Test',1, 3,1,gpu=False)
    torch.set_default_dtype(torch.float16)
    torch.set_default_tensor_type(torch.HalfTensor)
    policy = MADDPG_Policy('Train',0,3,1,gpu=True)
    policy.act(10000, 50, 10,5, ifRender=False, load=False)  # 训练轮次，每轮步数，每步Buffer里面挑选的个数，渲染，加载已保存得网络
    printReward(policy)
    # 运行这个命令来训练（目录：(base) maddpg@game-ai:~/v1/Car_Tracing$）：
    # nohup python -u MADDPG_Policy_astar.py 2>&1 &
    # 下载网络的命令，在本地cmd里：
    # scp -P 22 -r maddpg@10.16.29.94:/home/maddpg/v1/MPE/data_tag D:\data_set\data_new
