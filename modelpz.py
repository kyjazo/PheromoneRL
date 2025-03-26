from pettingzoo import ParallelEnv
import numpy as np
import gymnasium as gym
from mesa.space import MultiGrid
from model import WolfSheepModel
from agents import Wolf, Sheep, Pheromone
import json


class WolfSheepEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], "name": "wolf_sheep_v0"}

    def __init__(self, render_mode=None, width=20, height=20, initial_wolves=5, initial_sheep=20):
        super().__init__()
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.initial_wolves = initial_wolves
        self.initial_sheep = initial_sheep

        # Initialize the model
        self.model = WolfSheepModel(
            width=width,
            height=height,
            initial_wolves=initial_wolves,
            initial_sheep=initial_sheep,
            render_pheromone=False
        )

        # Setup agents
        self.possible_agents = [f"wolf_{i}" for i in range(initial_wolves)] + [f"sheep_{i}" for i in
                                                                               range(initial_sheep)]
        self.agents = self.possible_agents.copy()

        self.action_spaces = {agent: gym.spaces.Discrete(3) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: gym.spaces.Dict({
                'pheromones': gym.spaces.Box(low=0, high=1, shape=(2, width, height)),
                'agents': gym.spaces.Box(low=0, high=1, shape=(2, width, height))
        }) for agent in self.possible_agents
        }


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        # Reset the model
        self.model = WolfSheepModel(
            width=self.width,
            height=self.height,
            initial_wolves=self.initial_wolves,
            initial_sheep=self.initial_sheep,
            render_pheromone=False
        )

        self.agents = [f"wolf_{i}" for i in range(self.initial_wolves)] + [f"sheep_{i}" for i in
                                                                           range(self.initial_sheep)]
        self.possible_agents = self.agents.copy()

        observations = {agent_id: self._get_observation(agent_id) for agent_id in self.agents}
        infos = {agent_id: {} for agent_id in self.agents}

        return observations, infos

    def _get_observation(self, agent_id):
        wolf_pheromones = np.array(self.model.wolf_pheromone_layer.data)
        sheep_pheromones = np.array(self.model.sheep_pheromone_layer.data)

        wolf_positions = np.zeros((self.width, self.height))
        sheep_positions = np.zeros((self.width, self.height))

        for agent in self.model.agents:
            if isinstance(agent, Wolf):
                wolf_positions[agent.pos] = 1
            elif isinstance(agent, Sheep):
                sheep_positions[agent.pos] = 1

        return {
            'pheromones': np.stack([wolf_pheromones, sheep_pheromones]),
            'agents': np.stack([wolf_positions, sheep_positions])
        }

    def step(self, actions):

        for agent_id, action in actions.items():
            agent = self._get_agent_by_id(agent_id)
            if agent is not None:
                self._apply_action(agent, action)

        self.model.step()

        observations = {agent_id: self._get_observation(agent_id) for agent_id in self.agents}
        rewards = {agent_id: self._get_reward(agent_id) for agent_id in self.agents}

        terminated = {agent_id: False for agent_id in self.agents}
        truncated = {agent_id: False for agent_id in self.agents}
        infos = {agent_id: {} for agent_id in self.agents}

        if self.model.count_agents(Sheep) == 0:
            terminated = {agent_id: True for agent_id in self.agents}
            self.agents = []  # AGGIUNTA CHIAVE: Svuota la lista degli agenti attivi
            for agent_id in self.possible_agents:  # Usa possible_agents per infos
                infos[agent_id] = {'episode': {'r': rewards.get(agent_id, 0)}}
        else:
            for agent_id in list(self.agents):
                agent = self._get_agent_by_id(agent_id)
                if agent is None or (isinstance(agent, Sheep) and not agent.alive):
                    terminated[agent_id] = True
                    self.agents.remove(agent_id)

        return observations, rewards, terminated, truncated, infos

    def _get_agent_by_id(self, agent_id):

        agent_type, idx = agent_id.split('_')
        idx = int(idx)

        if agent_type == 'wolf':
            wolves = [a for a in self.model.agents if isinstance(a, Wolf)]
            return wolves[idx] if idx < len(wolves) else None
        else:
            sheep = [a for a in self.model.agents if isinstance(a, Sheep) and a.alive]
            return sheep[idx] if idx < len(sheep) else None

    def _apply_action(self, agent, action):

        if isinstance(agent, Wolf):

            possible_steps = self.model.grid.get_neighborhood(
                agent.pos,
                moore=True,
                include_center=False,
                radius=2 if agent.stayed else 1
            )

            pheromones = [
                Pheromone(
                    wolf_concentration=self.model.wolf_pheromone_layer.data[x, y],
                    sheep_concentration=self.model.sheep_pheromone_layer.data[x, y]
                ) for (x, y) in possible_steps
            ]

            if action == 0:
                best_steps = agent.get_best_step(possible_steps, pheromones, True)
                if best_steps:
                    self.model.grid.move_agent(agent, agent.random.choice(best_steps))
                agent.stayed = False
            elif action == 1:
                if possible_steps:
                    self.model.grid.move_agent(agent, agent.random.choice(possible_steps))
                agent.stayed = False
            elif action == 2:
                agent.stayed = not agent.stayed

            # Update Q-learning
            next_possible_steps = self.model.grid.get_neighborhood(
                agent.pos, moore=True, include_center=False
            )
            next_pheromones = [
                Pheromone(
                    wolf_concentration=self.model.wolf_pheromone_layer.data[x, y],
                    sheep_concentration=self.model.sheep_pheromone_layer.data[x, y]
                ) for (x, y) in next_possible_steps
            ]

            state = agent.q_learning.get_state(agent, pheromones)
            next_state = agent.q_learning.get_state(agent, next_pheromones)
            reward = agent.calculate_reward(False, agent.steps_since_last_capture)

            agent.q_learning.learn(state, action, reward, next_state)

        elif isinstance(agent, Sheep):
            possible_steps = self.model.grid.get_neighborhood(
                agent.pos, moore=True, include_center=False
            )

            pheromones = [
                Pheromone(
                    wolf_concentration=self.model.wolf_pheromone_layer.data[x, y],
                    sheep_concentration=self.model.sheep_pheromone_layer.data[x, y]
                ) for (x, y) in possible_steps
            ]

            best_steps = agent.get_best_step(possible_steps, pheromones, False)
            if best_steps:
                self.model.grid.move_agent(agent, agent.random.choice(best_steps))

    def _get_reward(self, agent_id):
        agent = self._get_agent_by_id(agent_id)
        if agent is None:
            return 0

        if isinstance(agent, Wolf):
            return agent.sheep_eaten * 10 - agent.steps_since_last_capture * 0.1
        elif isinstance(agent, Sheep):
            nearby_wolves = sum(
                1 for x, y in self.model.grid.get_neighborhood(agent.pos, moore=True)
                if any(isinstance(a, Wolf) for a in self.model.grid.get_cell_list_contents((x, y)))
            )
            return -nearby_wolves
        return 0

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step: {self.model.steps}")
            print(f"Wolves: {self.model.count_agents(Wolf)}")
            print(f"Sheep: {self.model.count_agents(Sheep)}")
            print(f"Total sheep eaten: {sum(w.sheep_eaten for w in self.model.agents if isinstance(w, Wolf))}")

            # Simple grid visualization
            grid = np.zeros((self.width, self.height), dtype=str)
            grid[:] = '.'

            for agent in self.model.agents:
                if isinstance(agent, Wolf):
                    grid[agent.pos] = 'W'
                elif isinstance(agent, Sheep) and agent.alive:
                    grid[agent.pos] = 'S'

            for row in grid:
                print(' '.join(row))
            print()

    def close(self):
        pass