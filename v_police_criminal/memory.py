import numpy as np

class Memory:
    def __init__(self, env, args):
        self.size = args.memory_size
        self.buffer = {}# a dict for different names
        for agent in env.world.agents:
            name = agent.name
            obs_shape = env.observation_spaces[name].shape[0]
            act_shape = env.action_spaces[name].shape[0]
            b = {}
            b['obs'] = np.empty((self.size, obs_shape), dtype=np.float32)
            b['action'] = np.empty((self.size, act_shape), dtype=np.float32)
            b['reward'] = np.empty(self.size, dtype=np.float32)
            b['obs_new'] = np.empty((self.size, obs_shape), dtype=np.float32)
            # b['image'] = np.empty((self.size, 3, env.world.size[0], env.world.size[1]), dtype=np.float32)
            self.buffer[name] = b
        # self.image_buffer = np.empty((self.size, 3, env.world.size[0], env.world.size[1]), dtype=np.float32)
        self.image_buffer = np.empty((self.size, 3, env.height, env.width), dtype=np.float32)
        self.temp = {}
        self.image_temp = None
        self.cur_idx = 0
        self.total_cnt = 0

    def add(self, name, key, data):
        if name not in self.temp: self.temp[name] = {}
        self.temp[name][key] = data

    #first reverse_nonorm then reverse_norm
    def add_image(self, data):
        # data = 255-data
        data = data/255
        self.image_temp = data
    
    def submit(self):
        for name in self.temp.keys():
            self.buffer[name]['obs'][self.cur_idx] = self.temp[name]['obs'].cpu()
            self.buffer[name]['action'][self.cur_idx] = self.temp[name]['action'].cpu()
            self.buffer[name]['reward'][self.cur_idx] = self.temp[name]['reward'].cpu()
            self.buffer[name]['obs_new'][self.cur_idx] = self.temp[name]['obs_new'].cpu()
            # self.buffer[name]['image'][self.cur_idx] = self.temp[name]['image']
        self.image_buffer[self.cur_idx] = self.image_temp.cpu()
        # print(self.image_temp.cpu())
        self.cur_idx = (self.cur_idx + 1) % self.size  # store in the cur_idx, First In First Out
        self.temp = {}
        self.image_temp = None
        self.total_cnt += 1

    def sample(self, batch_size):
        idx = np.random.choice(min(self.total_cnt, self.size), size=batch_size, replace=False)
        batch, images = {}, None
        for name in self.buffer.keys():
            buf = {key: self.buffer[name][key][idx] for key in self.buffer[name].keys()}
            batch[name] = buf
        images = self.image_buffer[idx]
        return batch, images
