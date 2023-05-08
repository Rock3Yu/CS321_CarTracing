import pygame
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
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
    def __init__(self, scenario, world, max_cycles, render_mode, local_ratio):
        super().__init__()

        self.render_mode = render_mode

        self.renderOn = False
        self.seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.local_ratio = local_ratio
        self.width = world.size[0]
        self.height = world.size[1]

        self.scenario.reset_world(self.world, self.np_random)

        self.agents          = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map      = {agent.name: idx for idx, agent in enumerate(self.world.agents)}

        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = {}
        self.observation_spaces = {}
        for agent in self.world.agents:
            self.action_spaces[agent.name] = spaces.Box(low = -1., high = 1., shape = (2,))
            self.observation_spaces[agent.name] = spaces.Box(low = -np.inf, high = np.inf, \
                shape = (4 + 2 * len(self.world.escape_pos) + agent.radar_num,))
        
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
    
    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
            )

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None: self.seed(seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

        for criminal in self.world.criminals:
            criminal.escaped = False
            criminal.captured = False
    
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        current_agent = self.agent_selection
        current_idx = self._index_map[current_agent]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for agent in self.agents:
                    self.truncations[agent] = True
                return True

        self._cumulative_rewards[current_agent] = 0
        self._accumulate_rewards()

        return all(
            criminal.escaped or criminal.captured
            for criminal in self.world.criminals
        )

    def _execute_world_step(self):
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            self._set_action(action, agent)
        
        self.world.step()

        global_reward_p = self.scenario.global_reward_p(self.world)
        for police in self.world.police:
            local_reward = self.scenario.local_reward_p(agent, self.world)
            self.rewards[police.name] = \
                global_reward_p * (1 - self.local_ratio) \
                + local_reward * self.local_ratio
        for criminal in self.world.criminals:
            local_reward = self.scenario.local_reward_c(agent, self.world)
            self.rewards[criminal.name] = local_reward

    def _set_action(self, action, agent):
        agent.action.u = action

    def render(self):
        if not self.renderOn:
            if self.render_mode == 'human':
                pygame.init()
                self.screen = pygame.display.set_mode(self.world.size)
            else:
                self.screen = pygame.Surface(self.world.size)
            self.renderOn = True
        
        self.draw()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        if self.render_mode == 'human':
            pygame.display.flip()
        return np.transpose(observation, axes=(2, 1, 0))

    def draw(self):
        self.screen.fill((255, 255, 255))

        pygame_block = self.world.landmarks[0].copy()
        pygame_block[:, 1] = self.world.size[1] - pygame_block[:, 1]
        pygame.draw.polygon(self.screen, (0, 0, 0), pygame_block, 1)
        
        for block in self.world.landmarks[1:]:
            pygame_block = block.copy()
            pygame_block[:, 1] = self.world.size[1] - pygame_block[:, 1]
            pygame.draw.polygon(self.screen, (0, 0, 0), pygame_block)

        for entity in self.world.entities:
            if isinstance(entity, Criminal) and (entity.escaped or entity.captured): continue

            x, y = entity.state.p_pos
            y = self.world.size[1] - y

            pygame.draw.circle(self.screen, entity.color, (x, y), entity.size)
            # pygame.draw.circle(self.screen, (0, 0, 0), (x, y), entity.size, 1)

            # if isinstance(entity, Agent):
            #     dx, dy = (0, 0) if entity.action.u is None else entity.action.u
            #     dx, dy = [dx, -dy] / (np.linalg.norm([dx, dy]) + 1e-6) * entity.size
            #     pygame.draw.circle(self.screen, (255, 255, 255), (x+dx, y+dy), entity.size/3)
            #     pygame.draw.circle(self.screen, (0, 0, 0), (x+x, y+dy), entity.size/3, 1)

    def close(self):
        if self.renderOn:
            pygame.display.quit()
            pygame.quit()
