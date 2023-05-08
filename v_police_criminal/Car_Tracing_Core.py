import numpy as np

class EntityState:
    def __init__(self):
        self.p_pos = None

class AgentState(EntityState):
    def __init__(self):
        super().__init__()
        self.p_vel = np.zeros(2)

class Action: 
    def __init__(self):
        self.u = np.zeros(2)

class Entity:
    def __init__(self):
        self.name = ''
        self.size = 1.
        self.color = np.zeros(3)
        self.state = EntityState()

class EscapePos(Entity):
    def __init__(self):
        super().__init__()

class Agent(Entity):
    def __init__(self):
        super().__init__()
        self.max_speed = 0.
        self.state = AgentState()
        self.action = Action()
        self.collided = False
        self.radar_num = 0
        self.radar_dis = None

class Police(Agent):
    def __init__(self):
        super().__init__()

class Criminal(Agent):
    def __init__(self):
        super().__init__()
        self.escaped = False
        self.captured = False

class World:
    def __init__(self):
        self.size = None

        self.police = None
        self.criminals = None
        self.landmarks = None
        self.escape_pos = None

        self.spawn_points = None

        self.dt = .1
        self.damping = .05

    @property
    def agents(self):
        return self.police + self.criminals
    
    @property
    def entities(self):
        return self.agents + self.escape_pos

    def step(self):
        for agent in self.agents:
            if isinstance(agent, Criminal) and (agent.escaped or agent.captured): continue
            agent.state.p_vel *= 1 - self.damping
            agent.state.p_vel += agent.action.u * self.dt
            speed = np.sqrt(np.sum(np.square(agent.state.p_vel)))
            if speed > agent.max_speed:
                agent.state.p_vel *= agent.max_speed / speed
            agent.collided = False
            self.move(agent)
            self.scan(agent)
        self.escape_cnt = 0
        self.capture_cnt = 0
        if isinstance(agent, Criminal):
            for ep in self.escape_pos:
                if np.linalg.norm(ep.state.p_pos - agent.state.p_pos) < agent.size + ep.size:
                    agent.escaped = True
                    self.escape_cnt += 1
            if not agent.escaped:
                for p in self.police:
                    if np.linalg.norm(p.state.p_pos - agent.state.p_pos) < agent.size + p.size:
                        agent.captured = True
                        self.capture_cnt += 1

    def move(self, agent):
        dis = np.linalg.norm(agent.state.p_vel * self.dt)
        mindis = np.inf
        for block in self.landmarks:
            for idx in range(len(block)):
                rdis = self.ray_distance(agent, 0, block[idx], block[0 if idx==len(block)-1 else idx+1])
                if rdis is not None: mindis = min(mindis, rdis)
        if mindis - agent.size < dis: agent.collided = True
        agent.state.p_pos += agent.state.p_vel * self.dt * min(mindis - agent.size, dis) / dis

    def scan(self, agent):
        agent.radar_dis.fill(np.inf)
        for i in range(agent.radar_num):
            rad = 2*np.pi * i/agent.radar_num
            for block in self.landmarks:
                for idx in range(len(block)):
                    rdis = self.ray_distance(agent, rad, block[idx], block[0 if idx==len(block)-1 else idx+1])
                    if rdis is not None: agent.radar_dis[i] = min(agent.radar_dis[i], rdis)
        if (agent.radar_dis < agent.size).any(): agent.collided = True
    
    def ray_distance(self, agent, rad, p1, p2):
        rx, ry = agent.state.p_vel
        if rx == ry == 0: rx = 1
        rx, ry = rx*np.cos(rad) - ry*np.sin(rad), rx*np.sin(rad) + ry*np.cos(rad)
        A_r, B_r, C_r = ry, -rx, rx * agent.state.p_pos[1] - ry * agent.state.p_pos[0]
        s1 = p1[0] * A_r + p1[1] * B_r + C_r
        s2 = p2[0] * A_r + p2[1] * B_r + C_r
        dis1, dis2 = p1 - agent.state.p_pos, p2 - agent.state.p_pos
        dir1 = (np.sign(dis1) == np.sign([rx, ry])).all()
        dir2 = (np.sign(dis2) == np.sign([rx, ry])).all()
        dis1, dis2 = np.linalg.norm(dis1), np.linalg.norm(dis2)
        if s1 == s2 == 0 and dir1 and dir2: return min(dis1, dis2)
        if s1 == 0 and dir1: return dis1
        if s2 == 0 and dir2: return dis2
        if s1 * s2 < 0:
            A_l, B_l, C_l = p1[1] - p2[1], p2[0] - p1[0], p1[0] * p2[1] - p2[0] * p1[1]
            intersect = np.linalg.solve([[A_r, B_r], [A_l, B_l]], [-C_r, -C_l])
            dis = intersect - agent.state.p_pos
            if (np.sign(dis) == np.sign([rx, ry])).all(): return np.linalg.norm(dis)
        return None
