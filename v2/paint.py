import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class csv_analyzer:
    def __init__(self, path) -> None:
        self.data = pd.read_csv(path)
        self.save_path = path[:-4] + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # all are 1-index
        self.epoch_num = len(set(self.data['epoch']))
        self.step_num = len(set(self.data['step']))
        self.agent_num = len(set(self.data['agent']))

    def analyze(self):
        print('Analyze data & draw tables.')
        plt.figure(figsize=(4, 3), dpi=120)
        self._speed_analyze()
        self._occupation_analyze()
        self._accurancy_analyze()
        
    def _speed_analyze(self):  # sourcery skip: extract-duplicate-method
        '''
        save:
            table1: for each agent, x-step, y-average of (1/1) occupation for epochs
            table2: for all  agent, x-step, y-average of (3/3) occupation for epochs
        '''
        # table1
        x = range(self.step_num)
        y0 = [val['occupy'] for idx, val in self.data.iterrows() if val['agent'] == 'agent_0']
        y0 = np.array(y0).reshape(-1, self.step_num).sum(axis=0) / self.epoch_num
        y1 = [val['occupy'] for idx, val in self.data.iterrows() if val['agent'] == 'agent_1']
        y1 = np.array(y1).reshape(-1, self.step_num).sum(axis=0) / self.epoch_num
        y2 = [val['occupy'] for idx, val in self.data.iterrows() if val['agent'] == 'agent_2']
        y2 = np.array(y2).reshape(-1, self.step_num).sum(axis=0) / self.epoch_num
        plt.figure(figsize=(4, 3), dpi=120)
        plt.plot(x, y0, color='r', linestyle='-', label = 'agent_0')
        plt.plot(x, y1, color='g', linestyle='--', label = 'agent_1')
        plt.plot(x, y2, color='b', linestyle='-.', label = 'agent_2')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        # plt.xticks(x[::5])
        plt.xlabel('Step')
        plt.ylabel('Averaget occupation rate')
        plt.title('Speed Analyzation, each agent')
        png_path = self.save_path + 'speed_analyze_each_agent.png'
        plt.savefig(png_path)
        print('Save to: ' + png_path)
        plt.close()
        
        # table2
        y_avg = (y0 + y1 + y2) / self.agent_num
        plt.figure(figsize=(4, 3), dpi=120)
        plt.plot(x, y_avg, color='r', linestyle='-', label = 'agent_0 1 2\'s average')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Averaget occupation rate')
        plt.title('Speed Analyzation, agents\' average')
        png_path = self.save_path + 'speed_analyze_agents_average.png'
        plt.savefig(png_path)
        print('Save to: ' + png_path)
        plt.close()

    def _occupation_analyze(self):  # sourcery skip: comprehension-to-generator
        '''
        save:
            bar-chart1: for each agent and average, the last step occupation
        '''
        x = range(self.agent_num + 1)
        y0 = sum([val['occupy'] for idx, val in self.data.iterrows() 
              if val['agent'] == 'agent_0' and val['step'] == self.step_num - 1]) / self.epoch_num
        y1 = sum([val['occupy'] for idx, val in self.data.iterrows() 
              if val['agent'] == 'agent_1' and val['step'] == self.step_num - 1]) / self.epoch_num
        y2 = sum([val['occupy'] for idx, val in self.data.iterrows() 
              if val['agent'] == 'agent_2' and val['step'] == self.step_num - 1]) / self.epoch_num
        y = [y0, y1, y2, (y0 + y1 + y2) / 3]
        color=['red', 'peru', 'orchid', 'deepskyblue']
        x_label=['agent_0','agent_1','agent_2','average']
        plt.xticks(x, x_label)
        plt.bar(x, y, color=color)
        plt.grid(True, linestyle='--', alpha=0.5)
        for a, b in zip(x, y):
            plt.text(a, b, '%.3f'%b, ha='center', va='bottom', fontsize=7);
        plt.ylabel('Occupation rate at the last step')
        plt.title('Occupation Analyzation')
        png_path = self.save_path + 'occupation_analyze.png'
        plt.savefig(png_path)
        print('Save to: ' + png_path)
        plt.close()
        
    def _accurancy_analyze(self):
        '''
        save:
            table1: agents' average, x-step, y-occupation accurancy for epochs
        '''
        x = range(self.step_num)
        y = [val['global_distance'] for idx, val in self.data.iterrows() if val['agent'] == 'agent_0']
        y = np.array(y).reshape(-1, self.step_num).sum(axis=0) / self.epoch_num
        plt.figure(figsize=(4, 3), dpi=120)
        plt.plot(x, y, color='r', linestyle='-', label = 'average')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.yticks(np.arange(-0.2, max(y), 0.2))
        plt.xlabel('Step')
        plt.ylabel('Distance')
        plt.title('Occupation Accurancy Analyzation')
        png_path = self.save_path + 'occupation_accurancy_analyze_average.png'
        plt.savefig(png_path)
        print('Save to: ' + png_path)
        plt.close()


if __name__ == '__main__':
    path = './log/spread_l7/benchmark.csv'
    tool = csv_analyzer(path)
    tool.analyze()