import numpy as np

from pettingzoo.mpe._mpe_utils.scenario import BaseScenario

from Car_Tracing_Core import EscapePos, Police, Criminal, World
from Car_Tracing_Env import CarTracingEnv, make_env

from map_manager import file2map

class raw_env(CarTracingEnv):
    def __init__(self, map_path, max_cycles, render_mode, local_ratio):
        scenario = CarTracingScenario()
        world = scenario.make_world(map_path)
        super().__init__(scenario, world, max_cycles, render_mode, local_ratio)
        self.metadata = {}
        self.metadata['name'] = 'car_tracing'

env = make_env(raw_env)

class CarTracingScenario(BaseScenario):
    def make_world(self, map_path):
        world = World()

        size, police, criminals, landmarks, spawn_points, escape_pos = file2map(map_path)

        world.size = size
        world.spawn_points = spawn_points

        world.police = [Police() for _ in range(len(police))]
        for i, p in enumerate(world.police):
            p.name = f'police_{i}'
            p.size = 1.5
            p.color = np.array([51, 51, 255])
            p.max_speed = 10.
            p.radar_num = police[i]
            p.radar_dis = np.zeros(p.radar_num)

        world.criminals = [Criminal() for _ in range(len(criminals))]
        for i, c in enumerate(world.criminals):
            c.name = f'criminal_{i}'
            c.size = 1.5
            c.color = np.array([255, 51, 51])
            c.max_speed = 10.5
            c.radar_num = criminals[i]
            c.radar_dis = np.zeros(c.radar_num)

        world.landmarks = landmarks

        world.escape_pos = [EscapePos() for _ in range(escape_pos)]
        for i, e in enumerate(world.escape_pos):
            e.name  = f'escape_pos_{i}'
            e.size  = 1.
            e.color = np.array([51, 204, 51])

        return world

    def reset_world(self, world, np_random:np.random.Generator):
        for agent in world.agents:
            agent.state.p_vel = np.zeros(2)
        
        spp = np_random.choice(world.spawn_points.shape[0], len(world.entities), replace=False)
        for i, entity in enumerate(world.entities):
            entity.state.p_pos = world.spawn_points[spp[i]]
        # print(world.spawn_points[spp])
        # exit()
        

    def benchmark_data(self, agent, world):
        return
    
    def local_reward_p(self, agent, world):
        r1 = 0
        for p in world.police:
            if p is agent: continue
            if np.linalg.norm(p.state.p_pos - agent.state.p_pos) < agent.size + p.size:
                agent.collided = True
                break
        if agent.collided:
            r1 = -10
            agent.collided = False

        distances_cri = [
                np.linalg.norm(agent.state.p_pos - cri.state.p_pos)
                for cri in world.criminals
            ]
        r2 = -min(distances_cri)

        r3 = (
            -5
            if any(
                np.linalg.norm(agent.state.p_pos - pol.state.p_pos)
                < agent.size * 5
                for pol in world.police
            )
            else 0
        )

        return r1 + r2 + r3
    
    def local_reward_c(self, agent, world):
        r1 = 0
        for c in world.criminals:
            if c is agent: continue
            if np.linalg.norm(c.state.p_pos - agent.state.p_pos) < agent.size + c.size:
                agent.collided = True
                break
        if agent.collided:
            r1 = -10
            agent.collided = False
        
        distances_esp= [
                np.linalg.norm(agent.state.p_pos - esp.state.p_pos)
                for esp in world.escape_pos
            ]
        distances_pol= [
                np.linalg.norm(agent.state.p_pos - pol.state.p_pos)
                for pol in world.police
            ]
        r2 = -min(distances_esp) + min(distances_pol)
        
        r3 = 500 if min(distances_esp) < agent.size + world.escape_pos[np.argmin(distances_esp)].size else 0

        return r1 + r2 + r3

    
    def global_reward_p(self, world):
        r2, r4 = 0, 0
        for cri in world.criminals:
            dists = [
                np.linalg.norm(pol.state.p_pos - cri.state.p_pos)
                for pol in world.police
            ]
            r2 -= min(dists)
            if min(dists) < cri.size + world.police[np.argmin(dists)].size: r4=500
        r2 /= len(world.criminals)
        
        return r2 + r4

    def observation(self, agent, world):
        return np.concatenate([
            agent.state.p_pos, 
            agent.state.p_vel,
            np.array([e.state.p_pos for e in world.escape_pos]).flatten(),
            agent.radar_dis
            ])
