import numpy as np

class Memory:
    def __init__(self, env, args):
        self.size = args.memory_size
        self.buffer = dict()
        for agent in env.world.agents:
            name = agent.name
            obs_shape = env.observation_spaces[name].shape[0]
            act_shape = env.action_spaces[name].shape[0]
            b = dict()
            b['obs'] = np.empty((self.size, obs_shape), dtype=np.float32)
            b['action'] = np.empty((self.size, act_shape), dtype=np.float32)
            b['reward'] = np.empty(self.size, dtype=np.float32)
            b['obs_new'] = np.empty((self.size, obs_shape), dtype=np.float32)
            self.buffer[name] = b
        self.temp = dict()
        self.cur_idx = 0
        self.total_cnt = 0

    def add(self, name, key, data):
        if name not in self.temp: self.temp[name] = dict()
        self.temp[name][key] = data
    
    def submit(self):
        for name in self.temp.keys():
            self.buffer[name]['obs'][self.cur_idx] = self.temp[name]['obs'].cpu()
            self.buffer[name]['action'][self.cur_idx] = self.temp[name]['action'].cpu()
            self.buffer[name]['reward'][self.cur_idx] = self.temp[name]['reward'].cpu()
            self.buffer[name]['obs_new'][self.cur_idx] = self.temp[name]['obs_new'].cpu()
        self.cur_idx = (self.cur_idx + 1) % self.size
        self.total_cnt += 1

    def sample(self, batch_size):
        idx = np.random.choice(min(self.total_cnt, self.size), size=batch_size, replace=False)
        batch = dict()
        for name in self.buffer.keys():
            buf = dict()
            for key in self.buffer[name].keys():
                buf[key] = self.buffer[name][key][idx]
            batch[name] = buf
        return batch
