import numpy as np

from pettingzoo.mpe._mpe_utils.core import Action

class EntityState:
    def __init__(self):
        self.p_pos = None

class AgentState(EntityState):
    def __init__(self):
        super().__init__()
        self.p_vel = None

class Entity:
    def __init__(self):
        self.name  = ''
        self.size  = 1.
        self.color = np.array([0., 0., 0.])
        self.state = EntityState()

class EscapePos(Entity):
    def __init__(self):
        super().__init__()

class Landmark(Entity):
    def __init__(self):
        super().__init__()

class Agent(Entity):
    def __init__(self):
        super().__init__()
        self.max_speed = 0.
        self.state     = AgentState()
        self.action    = Action()

class Police(Agent):
    def __init__(self):
        super().__init__()

class Criminal(Agent):
    def __init__(self):
        super().__init__()
        self.escape  = False
        self.capture = False

class World:
    def __init__(self):
        self.map = None

        self.size               = (0, 0)
        self.default_police     = []
        self.default_criminals  = []
        self.default_landmarks  = []
        self.default_escape_pos = []

        self.police     = []
        self.criminals  = []
        self.landmarks  = []
        self.escape_pos = []

        self.dt = 0.1

        self.damping = .05

    @property
    def agents(self):
        return self.police + self.criminals

    @property
    def solid_entities(self):
        return self.agents + self.landmarks
    
    @property
    def entities(self):
        return self.solid_entities + self.escape_pos

    def step(self, rewards):
        for _, agent in enumerate(self.agents):
            if isinstance(agent, Criminal) and (agent.capture or agent.escape):
                agent.state.p_vel = [0., 0.]
                continue

            agent.state.p_vel *= 1 - self.damping
            
            agent.state.p_vel += agent.action.u * agent.max_speed * self.dt * 5

            for _, entity in enumerate(self.agents):
                if agent is entity or (isinstance(entity, Criminal) and (entity.capture or entity.escape)):
                    continue
                delta_pos = entity.state.p_pos - agent.state.p_pos
                distance  = np.sqrt(np.sum(delta_pos ** 2))
                r_sum     = agent.size + entity.size
                if distance >= r_sum:
                    continue
                rewards[agent.name] = -10
                k = (1.3 * r_sum - distance) * .8
                agent.state.p_vel -= k * delta_pos / distance

            coor = agent.state.p_pos // 1
            for i in range(-2, 3):
                if coor[0]+i < 0 or coor[0]+i >= self.size[0]:
                    continue
                for j in range(-2, 3):
                    if coor[1]+j < 0 or coor[1]+j >= self.size[1]:
                        continue
                    if abs(i) == 2 and abs(j) == 2:
                        continue
                    coor_ob = (int(coor[0]+i), int(coor[1]+j))
                    if self.map[coor_ob[0]][coor_ob[1]] == 'o':
                        rewards[agent.name] = -10
                        delta_pos = coor_ob - agent.state.p_pos
                        distance  = np.sqrt(np.sum(delta_pos ** 2))
                        k = (1.6 * (agent.size + .5) - distance) * 2.5
                        agent.state.p_vel -= k * delta_pos / distance

            if agent.state.p_pos[0] - agent.size / 4 < 0:
                agent.state.p_vel[0] += 1.5 * agent.max_speed
            if agent.state.p_pos[0] + agent.size / 4 >= self.size[0]:
                agent.state.p_vel[0] -= 1.5 * agent.max_speed
            if agent.state.p_pos[1] - agent.size / 4 < 0:
                agent.state.p_vel[1] += 1.5 * agent.max_speed
            if agent.state.p_pos[1] + agent.size / 4 >= self.size[1]:
                agent.state.p_vel[1] -= 1.5 * agent.max_speed

            speed = np.sqrt(agent.state.p_vel[0] ** 2 + agent.state.p_vel[1] ** 2)
            if speed > agent.max_speed:
                agent.state.p_vel = agent.state.p_vel / speed * agent.max_speed
            agent.state.p_pos += agent.state.p_vel * self.dt

class Policy:
    def __init__(self, observation, reward, done, info):
        self.size = 3

        self.is_police      = observation[0]
        self.idx            = observation[1]
        self.p_pos          = observation[2]
        self.criminal_state = observation[3]
        self.police_pos     = observation[4]
        self.criminal_pos   = observation[5]
        self.landmark_pos   = observation[6]
        self.escape_pos     = observation[7]

        self.map = self.to_map_arr(observation[8])

        self.reward = reward
        self.done   = done
        self.info   = info

    def act(self):
        raise NotImplementedError()

    def to_map_arr(self, world_map):
        for pos in self.police_pos:
            world_map[int(pos[0]//1)][int(pos[1]//1)] = 'p'
        for pos in self.criminal_pos:
            world_map[int(pos[0]//1)][int(pos[1]//1)] = 'c'
        return world_map
