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
        pass