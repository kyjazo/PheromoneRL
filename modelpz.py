from pettingzoo import ParallelEnv
import numpy as np
import gymnasium.spaces
from mesa.space import MultiGrid

from model import WolfSheepModel, Wolf


class WolfSheepEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], "name": "wolf_sheep_v0"}

    def __init__(self):
        self.model = WolfSheepModel()
        self.agents = {agent.unique_id: agent for agent in self.model.agents}
        self.possible_agents = list(self.agents.keys())

        self.observation_spaces = {
            agent_id: gymnasium.spaces.Box(low=0, high=1, shape=(self.model.grid.width, self.model.grid.height),
                                           dtype=np.float32)
            for agent_id in self.possible_agents
        }

        self.action_spaces = {
            agent_id: gymnasium.spaces.Discrete(8)
            for agent_id in self.possible_agents
        }

    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def reset(self, seed=None, options=None):
        self.model = WolfSheepModel()
        self.agents = {agent.unique_id: agent for agent in self.model.agents}
        return {agent_id: self.get_obs(agent) for agent_id, agent in self.agents.items()}

    def get_obs(self, agent):
        obs = np.zeros((self.model.grid.width, self.model.grid.height), dtype=np.float32)
        obs[agent.pos] = 1
        return obs

    def step(self, actions):
        for agent_id, action in actions.items():
            agent = self.agents[agent_id]
            self.apply_action(agent, action)

        self.model.step()

        observations = {agent_id: self.get_obs(agent) for agent_id, agent in self.agents.items()}
        rewards = {agent_id: self.get_reward(agent) for agent_id in self.agents}
        dones = {agent_id: not self.model.running for agent_id in self.agents}
        infos = {agent_id: {} for agent_id in self.agents}

        return observations, rewards, dones, infos

    def apply_action(self, agent, action):
        if agent.pos is None:
            return

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        new_pos = tuple(np.add(agent.pos, directions[action]))

        if not self.model.grid.out_of_bounds(new_pos):
            self.model.grid.move_agent(agent, new_pos)

    def get_reward(self, agent):
        if isinstance(agent, Wolf):
            return agent.sheep_eaten
        return 0

    def render(self, mode="human"):
        print(self.model.grid)

    def close(self):
        pass
