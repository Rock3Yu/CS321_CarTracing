import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import Astar_Policy as astar_p

import Car_Tracing_Scenario

# dim=self.num_agent*2+self.num_escape*2+121+20*2=8+2+121+40=171
class MADDPG_Policy:
    def __init__(self, mode,num_police, num_criminal,num_escape,gpu=False):  # mode = Train,Test
        self.gpu=gpu
        self.mode=0 if mode=='Train' else 1  #'Test'=0 or 'Train'=1
        self.num_police = num_police
        self.num_criminal = num_criminal
        self.num_agent = num_police + num_criminal
        self.num_escape=num_escape
        self.done=[False]*self.num_agent
        self.is_criminal=[False] * self.num_agent
        self.actors = [None] * self.num_agent  # actor neural
        self.actors_target = [None] * self.num_agent # actor target neurual
        self.critics = [None] * self.num_agent  # critic neural
        self.critics_target = [None] * self.num_agent
        self.optimizer_critics = [None] * self.num_agent
        self.optimizer_actors = [None] * self.num_agent
        self.memory = self.Memory(100000)
        self.actionMemory=[[0., 0.] * 20] * self.num_agent
        self.tao = .01  # 学习率
        self.gamma = .97  # 贴现因子，考虑越长远，给现在的影响越小
        self.reward_record=[]

        for i in range(self.num_agent):
            self.actors[i] = self.Actor()
            self.actors_target[i] = self.Actor()
            self.critics[i] = self.Critic()
            self.critics_target[i] = self.Critic()
            self.optimizer_actors[i] = optim.Adam(self.actors[i].parameters(), lr=1.)  # 学习率1.直接更新，momentum动量
            self.optimizer_critics[i] = optim.Adam(self.critics[i].parameters(), lr=1.)
            self.actors_target[i].load_state_dict(self.actors[i].state_dict())
            self.critics_target[i].load_state_dict(self.critics[i].state_dict())

    def act(self,num_epoch, cycles, train_batch,ifRender): 
        if self.gpu:
            self.importcuda()
        # env = Car_Tracing_Scenario.env(map_path = 'map/test_map_d.txt', max_cycles = cycles, ratio = .9)  # 4
        # env = Car_Tracing_Scenario.env(map_path = 'map/test_map.txt', max_cycles = cycles, ratio = 1.5)  # 4
        env = Car_Tracing_Scenario.env(map_path = 'map/city.txt', max_cycles = cycles, ratio = .9)#4
        for i in range(num_epoch):
            self.reward_record.append([]) #加入每一轮的结果

        # self.load()
        for epoch in range(num_epoch):  # the epoch is a training process 1个epoch等于使用训练集中的全部样本训练一次，通俗的讲epoch的值就是整个数据集被轮几次。
            env.reset()
            self.done=[False]*self.num_agent
            print('epoch',epoch)
            for step in range(cycles):  # the step is a step in max-steps
                # print(f'step:{step}:',self.done)
                action_idx = step % 20
                if (step+1) % 50 == 0:
                    print(f'training...  step={step+1}/{cycles}' if self.mode==0 else f'testing...  step={step+1}/{cycles}')
                actions = []
                done_cnt=0
                if ifRender:
                    env.render()
                obs_agents = self.getAgentObs(env)   #get all agents' obs seperately, given obs
                for i in range(self.num_agent):
                    # TODO:计算真正的obs，计算视野范围，加动作序列
                    if 'criminal' in env.last()[0]['name']:
                        self.is_criminal[i]=True
                        # print(env.last()[0]['rew'])
                        if abs(env.last()[0]['rew'][0])==250 or self.done[i]:
                            self.done[i]=True
                            action=np.array([0.,0.])
                            done_cnt+=1
                        else:
                            action=self.ToAstar(env.last()[0]['obs'],env.last()[0]['map'],i)
                    else: #police
                        if  self.gpu:
                            action = self.actors[i](torch.as_tensor( obs_agents[i],device='cuda:0',dtype=torch.float)).detach().cpu().numpy() 
                        else:
                            action = self.actors[i](torch.as_tensor(obs_agents[i],dtype=torch.float)).detach().numpy()
                    
                            # 即返回一个新的tensor，从当前计算图中分离下来。 但是仍指向原变量 的 存放位置，不能用于计算梯度
                    action = action/np.max(np.abs(action)) if np.max(np.abs(action))!=0. else np.array([0.,0.])  #normalize
                    
                    if not self.is_criminal[i]:
                        astar_a=self.ToAstar(env.last()[0]['obs'],env.last()[0]['map'],i)
                        astar_a=astar_a/np.max(np.abs(action));astar_a=np.nan_to_num(astar_a)
                        action=np.nan_to_num(action)+np.random.normal(0,1,2)+astar_a/20
                        action = action/np.max(np.abs(action)) if np.max(np.abs(action))!=0. else np.array([0.,0.])  #normalize
                    else:
                        action= np.nan_to_num(action)
                    self.actionMemory[i][action_idx] = action  #更新记忆动作序列
                    env.step(action)
                    if self.mode==0:  # Train#################################
                        actions.append(action)
                rewards = env.last()[0]['rew']  # all agents' reward
                action_idx = (action_idx+1)%20  # update action idx
                ###################
                self.reward_record[epoch].append(rewards)
                ###################


                if done_cnt== self.num_criminal: #if all criminal is catched
                    print(done_cnt,self.num_criminal)
                    print(f'epoch {epoch} ends at step {step+1}')
                    break
                # if self.mode==1:#Test
                #     obs= np.array(env.last()[0]['obs'])
                if self.mode==0:  # Train#########################################
                    #TODO:计算真正的obs_new，计算视野范围，加动作序列
                    obs_new_agents = self.getAgentObs(env)                    
                    self.memory.add(obs_agents, actions, rewards, obs_new_agents)  #the buffer for all agents
                    
                    if step > 19:
                        for cnt in range(train_batch):#训练的batch是数目，每次选择多少条sample训练
                            for i in range(0,self.num_agent-1):
                                if self.is_criminal[i]:
                                    continue
                                obs_agents, actions, rewards, obs_new_agents = self.memory.sample() #随机选择buffer中的一组数据
                                obs=obs_agents[i];obs_new=obs_new_agents[i]
                                actions_new = []
                                for j in range(0,self.num_agent):
                                    if j==self.num_agent-1:
                                        action_new=self.ToAstar(env.last()[0]['obs'],env.last()[0]['map'],i)
                                    else:
                                        if  self.gpu:
                                            action_new = self.actors_target[i](torch.as_tensor(obs_new,device='cuda:0',dtype=torch.float)).detach().cpu().numpy() 
                                        else:
                                            action_new = self.actors_target[j](torch.as_tensor((obs_new),dtype=torch.float)).detach().numpy() #对每个智体确定他们的下一步动作,根据A_target,μ'
                                    action_new = action_new/np.max(np.abs(action_new)) if np.max(np.abs(action))!=0. else np.array([0.,0.]) #将动作normalize归一化
                                    if not self.is_criminal[i]:
                                        action_new=np.nan_to_num(action_new)
                                        action_new = action_new/np.max(np.abs(action_new)) if np.max(np.abs(action_new))!=0. else np.array([0.,0.])  #normalize
                                    else:
                                        action_new=np.nan_to_num(action_new)
                                    # action_new=np.nan_to_num(action)+np.random.normal(0, 1, 2)
                                    # action_new = action_new/np.max(np.abs(action_new)) if np.max(np.abs(action))!=0. else np.array([0.,0.])
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
            if self.mode==0:
                self.save()
    def to1D(self,list2):
        list1=[]
        for t in list2:
            list1.append(t[0])
            list1.append(t[1])
        return list1 
    def getAgentObs(self,env):
        obs_agents=[]
        obs=env.last()[0]['obs']
        for i in range(self.num_agent):    #first is police 
            #TODO:计算真正的obs，计算视野范围，加动作序列
            obs_agenti=[]#for each agent
            obs_agenti.append(obs)  #list 
            scope=self.getScope(i,obs,env.last()[0]['map'])#return an 1d list of len 121 of  33*33 area
            obs_agenti.extend(scope)
            # mem=self.to1D(self.actionMemory[i])#action memory  需要这里处理一下将obsagenti降维到1d
            mem=self.actionMemory[i]
            obs_agenti.extend(mem)
            # obs_agenti= np.concatenate((np.array(obs_agenti[0]),np.array(obs_agenti[1])
            # ,np.array(obs_agenti[2])),axis=0) #转化为ndarray加速运算
            obs_agenti=np.ndarray((171,),dtype=float)
            obs_agents.append(obs_agenti) 
        return obs_agents
                    

    def ToAstar(self,obs,map,idx):
        # start=time.time()
        isPolicy=[True]*self.num_police+[False]*self.num_criminal
        idx=self.num_police  #which is criminal
        p_pos=((obs[idx*2]),(obs[idx*2+1]))#criminal position
        criminal_state=True
        police_pos=[]
        criminal_pos=[]
        landmark_pos=[]
        escape_pos=[]
        map_height=200
        map_width=250
        for i in range(len(obs)//2) :
            i=i*2
            if i< self.num_police*2:
                if obs[i]<0 or obs[i]>=map_height or obs[i+1]<0 or obs[i+1]>=map_width:
                    return np.array([0.,0.])
                map[int(obs[i]),int(obs[i+1])]='p'
                police_pos.append(((obs[i]),(obs[i+1])))
            elif i<self.num_agent*2:
                if  obs[i]<0 or obs[i]>=map_height or obs[i+1]<0 or obs[i+1]>=map_width:
                    return np.array([0.,0.])
                map[int(obs[i]),int(obs[i+1])]='c'
                criminal_pos.append(((obs[i]),(obs[i+1])))
            else:
                escape_pos.append(((obs[i]),(obs[i+1])))
        result = []
        result.append(isPolicy)
        result.append(idx)
        result.append(np.array(p_pos))
        result.append(criminal_state)
        result.append(np.array(police_pos))
        result.append(np.array(criminal_pos))
        result.append(landmark_pos)
        if idx==self.num_police:
            result.append(np.array(escape_pos))
        else:
            result.append(np.array(criminal_pos[0]))  #police is on criminal
        result.append(map)
        # mid=time.time()
        a_policy=astar_p.astarPolicy(result)
        action = a_policy.act()
        # end=time.time()
        # print('cost at prepare:',mid-start,', cost at astar:',end-mid)
        return np.array(action,dtype=float)
    def getScope(self,idx,obs,map):
        p_pos=(obs[idx*2],obs[idx*2+1]) 
        p_pos=(int(p_pos[0]),int(p_pos[1]))
        zero_one_table=np.zeros((121,),dtype=float)
        map_height=200
        map_width=250
        for i in range(-5,6):
            for j in range(-5,6):
                have_land=False
                for ind in range(9):
                    k=ind//3 -1
                    l=ind%3 -1
                    if p_pos[0]+i*3+k >=map_height or  p_pos[1]+j*3+l>=map_width\
                    or p_pos[0]+i*3+k <0 or  p_pos[1]+j*3+l<0:
                        have_land=True
                        break
                    elif map[p_pos[0]+i*3+k,p_pos[1]+j*3+l]=='o':
                        have_land=True
                        break
                if have_land:
                    zero_one_table[(i+5)*11+j+5]=1
        zero_one_table=zero_one_table.tolist()
        return zero_one_table

    def save(self):
        for i in range(self.num_agent):
            print(f'saving...  idx={i}')
            torch.save(self.actors[i].state_dict(), f'./data/actors/{i}')
            torch.save(self.actors_target[i].state_dict(), f'./data/actors_target/{i}')
            torch.save(self.critics[i].state_dict(), f'./data/critics/{i}')
            torch.save(self.critics_target[i].state_dict(), f'./data/critics_target/{i}')

    def load(self):
        path='./data/'
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
        def __init__(self):
            super().__init__()
            # self.fc1 = nn.Linear(171 , 171 )
            # self.fc2 = nn.Linear(171, 85)
            # self.fc3 = nn.Linear(85, 2)#####
            self.fc1 = nn.Linear(171 , 128 )
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 2)#####
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
            # self.fc1_1 = nn.Linear(171, 171)
            # self.fc1_2 = nn.Linear(171, 85)
            # self.fc2_1 = nn.Linear(4 * 2, 16)
            # self.fc2_2 = nn.Linear(16, 8)
            # self.fc3_1 = nn.Linear(93, 93*2)
            # self.fc3_2 = nn.Linear(93*2, 93)
            # self.fc3_3 = nn.Linear(93, 1) ############
            self.fc3_1 = nn.Linear(171+8, 128)
            self.fc3_2 = nn.Linear(128, 128)
            self.fc3_3 = nn.Linear(128, 1)
            self.relu = nn.ReLU()

        def forward(self, obs,action):
            obs=obs.flatten()
            action=action.flatten()
            # obs = self.fc1_1(obs)
            # obs = self.relu(obs)
            # obs = self.fc1_2(obs)
            # obs = self.relu(obs)
            # action = self.fc2_1(action)
            # action = self.relu(action)
            # action = self.fc2_2(action)
            # action = self.relu(action)
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
def printReward(policy):
    print_log = open("/home/maddpg/v1/Car_Tracing/reward.txt",'w')
    print(str(policy.reward_record),file = print_log)
    print_log.close()  # output reward to file，array of len num_epoch
if __name__ == '__main__':
    policy = MADDPG_Policy('Train',3, 1,1,gpu=True)
    # policy = MADDPG_Policy('Test',3, 1,1,gpu=False)
    policy.act(1000, 200, 5,ifRender=False)  # 训练轮次，每轮步数，每步Buffer里面挑选的个数，渲染
    printReward(policy)
    # 运行这个命令来训练（目录：(base) maddpg@game-ai:~/v1/Car_Tracing$）：
    # nohup python -u MADDPG_Policy_astar.py 2>&1 &
    # 下载网络的命令，在本地cmd里：
    # scp -P 22 -r maddpg@10.16.29.94:/home/maddpg/v1/Car_Tracing/data_100 D:\data_set\data_new

