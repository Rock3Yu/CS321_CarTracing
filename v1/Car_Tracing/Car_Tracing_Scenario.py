import numpy as np

from pettingzoo.mpe._mpe_utils.scenario import BaseScenario

from Car_Tracing_Core import EscapePos, Police, Criminal, Landmark, World
from Car_Tracing_Env import CarTracingEnv, make_env

from map_manager import file2map

class raw_env(CarTracingEnv):
    def __init__(self, map_path, max_cycles, ratio):
        scenario = CarTracingScenario()
        world    = scenario.make_world(map_path)
        super().__init__(scenario, world, max_cycles, ratio)

env = make_env(raw_env)

class CarTracingScenario(BaseScenario):
    def make_world(self, map_path):
        world = World()

        map, police, criminals, landmarks, escape_pos = file2map(map_path)

        world.map = np.array(map, dtype=str)
        for arr in world.map:
            for ch in arr:
                if ch != 'o' and ch !='e':
                    ch = ''

        world.size               = (len(map), len(map[0]))
        world.default_police     = police
        world.default_criminals  = criminals
        world.default_landmarks  = landmarks
        world.default_escape_pos = escape_pos

        world.police = [Police() for _ in range(len(police))]
        for i, p in enumerate(world.police):
            p.name      = f'police_{i}'
            p.size      = 1.5
            p.color     = np.array([.2, .2, 1])
            p.max_speed = 10.

        world.criminals = [Criminal() for _ in range(len(criminals))]
        for i, c in enumerate(world.criminals):
            c.name      = f'criminal_{i}'
            c.size      = 1.5
            c.color     = np.array([1, .2, .2])
            c.max_speed = 10.5

        world.landmarks = [Landmark() for _ in range(len(landmarks))]
        for i, o in enumerate(world.landmarks):
            o.name  = f'landmark_{i}'
            o.size  = .5
            o.color = np.array([.2, .2, .2])

        world.escape_pos = [EscapePos() for _ in range(len(escape_pos))]
        for i, e in enumerate(world.escape_pos):
            e.name  = f'escape_pos_{i}'
            e.size  = 1.
            e.color = np.array([.2, .8, .2])

        return world

    def reset_world(self, world, np_random):
        for i, police in enumerate(world.police):
            police.state.p_pos = np.array(world.default_police[i])
            police.state.p_vel = np.array([0., 0.])

        for i, criminal in enumerate(world.criminals):
            criminal.state.p_pos = np.array(world.default_criminals[i])
            criminal.state.p_vel = np.array([0., 0.])

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array(world.default_landmarks[i])

        for i, escape_pos in enumerate(world.escape_pos):
            escape_pos.state.p_pos = np.array(world.default_escape_pos[i])

    def observation(self, agent, world, rew):
        escape_pos = []
        for escape in world.escape_pos:
            escape_pos.append(escape.state.p_pos)

        landmark_pos = []
        for landmark in world.landmarks:
            landmark_pos.append(landmark.state.p_pos)

        police_pos = []
        for police in world.police:
            police_pos.append(police.state.p_pos)
        
        criminal_pos   = []
        for criminal in world.criminals:
            if not criminal.capture and not criminal.escape:
                criminal_pos.append(criminal.state.p_pos)
            else:
                criminal_pos.append((0., 0.))

        result = {}
        obs = []
        for p in police_pos:
            obs.append(p[0])
            obs.append(p[1])
        for p in criminal_pos:
            obs.append(p[0])
            obs.append(p[1])
        for p in escape_pos:
            obs.append(p[0])
            obs.append(p[1])
        result['obs']  = obs
        result['rew']  = list(rew.values())
        result['name'] = agent.name
        result['map']  = world.map  #已经转化成ndarray:str
        return result
