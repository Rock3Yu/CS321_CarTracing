import numpy as np

import pygame
from gym import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from Car_Tracing_Core import Agent, Criminal

def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        env = wrappers.ClipOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env
    
    return env

class CarTracingEnv(AECEnv):
    def __init__(self, scenario, world, max_cycles, render_mode, ratio=0.5):
        super().__init__()

        pygame.init()
        self.ratio  = ratio * 4
        self.width  = world.size[0] * 4 * ratio
        self.height = world.size[1] * 4 * ratio
        self.screen = pygame.Surface([self.height, self.width])

        self.renderOn = False

        self.metadata = {
            'render_modes' : 'human'
        }

        self.max_cycles = max_cycles
        self.scenario   = scenario
        self.world      = world

        self.scenario.reset_world(self.world, None)

        self.agents          = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map      = {agent.name: idx for idx, agent in enumerate(self.world.agents)}

        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = {}
        self.observation_spaces = {}
        for agent in self.world.agents:
            self.action_spaces[agent.name] = spaces.Box(low = -1., high = 1., shape = (2,))
            self.observation_spaces[agent.name] = spaces.Box(low = -np.inf, high = np.inf, shape = (0))
            raise ValueError('shape of observation not calculated yet')
        
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world, self.rewards
            )

    def reset(self, seed=None, return_info=False, options=None):
        self.scenario.reset_world(self.world, None)

        self.agents              = self.possible_agents[:]
        self.rewards             = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.dones               = {name: False for name in self.agents}
        self.infos               = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps           = 0

        self.current_actions = [None] * self.num_agents
        self.cur_agent = None

        for a in self.world.agents:
            if 'criminal' in a.name:
                a.escape = False
                a.capture = False
    
    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        current_agent        = self.agent_selection
        self.cur_agent = current_agent
        current_idx          = self._index_map[current_agent]
        next_idx             = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for agent in self.agents:
                    self.dones[agent] = True

        self._cumulative_rewards[current_agent] = 0
        self._accumulate_rewards()

    def _execute_world_step(self):
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            self._set_action(action, agent)
        
        for key in self.rewards.keys():
            self.rewards[key] = 0
        self.world.step(self.rewards)

        for i, criminal in enumerate(self.world.criminals):
            if criminal.capture or criminal.escape:
                continue
            for escape in self.world.escape_pos:
                delta_pos = criminal.state.p_pos - escape.state.p_pos
                distance  = np.sqrt(np.sum(delta_pos ** 2))
                r_sum     = criminal.size + escape.size
                if distance < r_sum:
                    criminal.escape = True
                    for key in self.rewards.keys():
                        if 'police' in key:
                            self.rewards[key] += -250
                        else:
                            self.rewards[key] = 250
                        
                    print(f'{criminal.name} escaped at step {self.steps}')

        for i, police in enumerate(self.world.police):
            for criminal in self.world.criminals:
                if criminal.capture or criminal.escape:
                    continue
                delta_pos = police.state.p_pos - criminal.state.p_pos
                distance  = np.sqrt(np.sum(delta_pos ** 2))
                r_sum     = police.size + criminal.size
                if distance < r_sum:
                    criminal.capture = True
                    for key in self.rewards.keys():
                        if 'police' in key:
                            self.rewards[key] += 250
                        else:
                            self.rewards[key] += -250
                    print(f'{criminal.name} is captured at step {self.steps}')

    def _set_action(self, action, agent):
        agent.action.u = action

    def enable_render(self):
        if not self.renderOn:
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self, mode):
        self.enable_render()
        self.draw()
        pygame.display.flip()

    def draw(self):
        self.screen.fill((255, 255, 255))

        for _, entity in enumerate(self.world.entities):
            if isinstance(entity, Criminal) and entity.escape:
                continue

            y, x = entity.state.p_pos
            x *= self.ratio
            y *= self.ratio

            pygame.draw.circle(self.screen, entity.color * 255, (x, y), entity.size * self.ratio)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), entity.size * self.ratio, 1)

            if isinstance(entity, Agent):
                if isinstance(entity, Criminal) and entity.capture:
                    pygame.draw.line(self.screen, (0, 0, 0), 
                    (x - entity.size * self.ratio, y - entity.size * self.ratio), 
                    (x + entity.size * self.ratio, y + entity.size * self.ratio), 2)
                    pygame.draw.line(self.screen, (0, 0, 0), 
                    (x - entity.size * self.ratio, y + entity.size * self.ratio), 
                    (x + entity.size * self.ratio, y - entity.size * self.ratio), 2)
                else:
                    if entity.action.u is None or (entity.action.u == 0.).all():
                        _x, _y = 0., 0.
                    else:
                        _y, _x = entity.action.u
                        l = np.sqrt(_x ** 2 + _y ** 2)
                        _x, _y = _x / l * entity.size * 4, _y / l * entity.size * 4
                    pygame.draw.circle(self.screen, (255, 255, 255), (x+_x, y+_y), entity.size / 3 * self.ratio)
                    pygame.draw.circle(self.screen, (0, 0, 0), (x+_x, y+_y), entity.size / 3 * self.ratio, 1)

        pygame.draw.rect(self.screen, (0, 0, 0), ((0, 0), (self.height, self.width)), 1)


    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False