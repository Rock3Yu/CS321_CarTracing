import numpy as np
from Car_Tracing_Core import Policy

# class testPolicy(Policy):
    # def __init__(self, observation, reward, done, info):
    #     super().__init__(observation, reward, done, info)

    # def act(self):
    #     if self.is_police:
    #         criminal_idx = -1
    #         for i in range(len(self.criminal_state)):
    #             if not self.criminal_state[i]:
    #                 criminal_idx = i
    #                 break
    #         if criminal_idx == -1:
    #             delta_pos = np.array([0., 0.])
    #         else:
    #             delta_pos = self.criminal_pos[criminal_idx] - self.p_pos
    #     else:
    #         if self.criminal_state[self.idx]:
    #             delta_pos = np.array([0., 0.])
    #         else:
    #             delta_pos = self.escape_pos[0] - self.p_pos
    #     if not (delta_pos == 0.).all():
    #         delta_pos /= np.max(np.abs(delta_pos))
    #     return delta_pos
import heapq

class astarPolicy(Policy):
    def __init__(self, observation, reward=None, done=None, info=None, path=None, iter=None):
        super().__init__(observation, reward, done, info)
        self.pre_path = path
        self.iter     = iter

    def act(self):

        '''
        if self.iter % 40 != 0 and len(self.pre_path) > 1:
        #if self.pre_path is not None and len(self.pre_path) > 1 :
            path = self.pre_path[1:]
        else:
        '''
        path = self.astar(self.escape_pos[0]//1)
                
        delta_pos = path[0] - self.p_pos

        if (delta_pos != 0.).any():
            delta_pos /= np.max(np.abs(delta_pos))
        
        return delta_pos
    
    def astar(self, target):
        cnt = 0

        visited = [[False]*self.map.shape[1] for _ in range(self.map.shape[0])]
        checked = [[False]*self.map.shape[1] for _ in range(self.map.shape[0])]
        parent  = [[None]*self.map.shape[1] for _ in range(self.map.shape[0])]
        G       = [[None]*self.map.shape[1] for _ in range(self.map.shape[0])]
        H       = [[None]*self.map.shape[1] for _ in range(self.map.shape[0])]
        pq      = []
        path    = []

        G[int(self.p_pos[0]//1)][int(self.p_pos[1]//1)] = 0.
        H[int(self.p_pos[0]//1)][int(self.p_pos[1]//1)] = np.sqrt(np.sum((self.p_pos//1 - target) **  2))
        visited[int(self.p_pos[0]//1)][int(self.p_pos[1]//1)] = True
        heapq.heappush(pq, (H[int(self.p_pos[0]//1)][int(self.p_pos[1]//1)], cnt, self.p_pos//1))
        cnt += 1
        
        while len(pq) > 0:
            cur = heapq.heappop(pq)
            if checked[int(cur[2][0])][int(cur[2][1])]:
                continue
            checked[int(cur[2][0])][int(cur[2][1])] = True

            for di in range(-1, 2):
                if di + cur[2][0] < 0 or di + cur[2][0] >= self.map.shape[0]:
                    continue
                for dj in range(-1, 2):
                    if dj + cur[2][1] < 0 or dj + cur[2][1] >= self.map.shape[1]:
                        continue
                    if (cur[2]+[di, dj] == target).all():
                        path.append(target)
                        
                        tmp = cur[2]
                        ##############added
                        if type(tmp)==None:  
                            print('cur',cur)
                            return self.p_pos
                            #########################exception cur[2] is None: then not move
                        path.append(tmp)
                        while (parent[int(tmp[0])][int(tmp[1])] != self.p_pos//1).any():
                            tmp = parent[int(tmp[0])][int(tmp[1])]
                            path.append(tmp)
                        path.reverse()
                        return path
            
            for di in range(-1, 2):
                if di + cur[2][0] < 0 or di + cur[2][0] >= self.map.shape[0]:
                    continue
                for dj in range(-1, 2):
                    if dj + cur[2][1] < 0 or dj + cur[2][1] >= self.map.shape[1]:
                        continue
                    if di == 0 and dj == 0:
                        continue
                    nei = np.array([int(cur[2][0]) + di, int(cur[2][1]) + dj])

                    flag = False
                    for _i in range(-1, 2):
                        if _i + nei[0] < 0 or _i + nei[0] >= self.map.shape[0]:
                            continue
                        for _j in range(-1, 2):
                            if _j + nei[1] < 0 or _j + nei[1] >= self.map.shape[1]:
                                continue
                            if abs(_i) == 1 and abs(_j) == 1:
                                continue
                            if self.map[_i + nei[0]][_j + nei[1]] == 'o':
                                flag = True
                    if flag:
                        continue

                    _G = np.sqrt(di ** 2 + dj ** 2) + G[int(cur[2][0])][int(cur[2][1])]
                    _H = np.sqrt(np.sum((nei - target) ** 2))
                    if not visited[nei[0]][nei[1]]:
                        G[nei[0]][nei[1]] = _G
                        H[nei[0]][nei[1]] = _H
                        visited[nei[0]][nei[1]] = True
                        parent[nei[0]][nei[1]] = cur[2]
                        heapq.heappush(pq, (_G + _H, cnt, nei))
                        cnt += 1
                    else:
                        if G[nei[0]][nei[1]] + H[nei[0]][nei[1]] > _G + _H:
                            G[nei[0]][nei[1]] = _G
                            H[nei[0]][nei[1]] = _H
                            parent[nei[0]][nei[1]] = cur[2]
                            checked[nei[0]][nei[1]] = False
                            heapq.heappush(pq, (_G + _H, cnt, nei))
                            cnt += 1
        path.append(self.p_pos)
        return path

