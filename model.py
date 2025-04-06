import mesa
import json
from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
from agents import Wolf, Sheep, Pheromones, QLearning
import numpy as np


with open("config.json", "r") as f:
    config = json.load(f)


class WolfSheepModel(Model):

    def __init__(self, width=config["grid_width"], height=config["grid_height"], initial_wolves=config["num_wolves"],
                 initial_sheep=config["num_sheep"], pheromone_evaporation=config["pheromone_evaporation"],
                 pheromone_added=config["pheromone_added"], render_pheromone=False,  q_table_file="q_table.json", max_steps=100,
                 diffusion_rate=0.1, seed=None):
         super().__init__(seed=seed)
         self.diffusion_rate = diffusion_rate
         self.max_steps = max_steps
         self.q_table_file = q_table_file
         self.render_pheromone = render_pheromone
         self.wolf_pheromone_layer = PropertyLayer("wolf_pheromone_layer", height=height, width=width,
                                                   default_value=0.0)
         self.sheep_pheromone_layer = PropertyLayer("sheep_pheromone_layer", height=height, width=width,
                                                    default_value=0.0)
         self.pheromone_evaporation = pheromone_evaporation
         self.pheromone_added = pheromone_added
         self.grid = MultiGrid(width, height, torus=True)
         self.running = True

         self.datacollector = DataCollector(
             model_reporters={
                 "Steps": self.get_steps,
                 "Avg_Reward": lambda m: float(np.mean(
                     [float(w.last_reward) for w in m.agents
                      if isinstance(w, Wolf) and hasattr(w, 'last_reward')] or [0.0]))
             },
             agent_reporters={
                 "Sheep_eaten": lambda a: int(a.sheep_eaten) if hasattr(a, 'sheep_eaten') else None,
                  "Reward": lambda a: float(np.mean(a.rewards))  # Usa la media delle reward
                  if hasattr(a, 'rewards') and a.rewards else None
             }
         )

         self.place_agents(Wolf, initial_wolves)
         self.place_agents(Sheep, initial_sheep)
         if render_pheromone:
             pheromones = Pheromones.create_agents(model=self, n=width * height)
             positions = np.array(np.meshgrid(np.arange(width), np.arange(height))).T.reshape(-1, 2)
             for agent, pos in zip(pheromones, positions):
                 self.grid.place_agent(agent, tuple(pos))

    def place_agents(self, agent_class, num_agents):

        agents = agent_class.create_agents(model=self, n=num_agents)
        positions = np.column_stack((self.rng.integers(0, self.grid.width, size=num_agents),
                                     self.rng.integers(0, self.grid.height, size=num_agents)))
        for agent, pos in zip(agents, positions):
            self.grid.place_agent(agent, tuple(pos))

    def save_q_tables(self):
        if not hasattr(self, 'q_table_file') or not self.q_table_file:
            return

        q_tables = {}
        for agent in self.agents:
            if isinstance(agent, Wolf):
                for state, actions in agent.q_learning.q_table.items():
                    if state not in q_tables:
                        q_tables[state] = actions
                    else:
                        for action, value in actions.items():
                            if action in q_tables[state]:
                                q_tables[state][action] = (q_tables[state][action] + value) / 2
                            else:
                                q_tables[state][action] = value

        if q_tables:
            temp_q_learning = QLearning(actions=[0, 1, 2])
            temp_q_learning.q_table = q_tables
            temp_q_learning.save_q_table(self.q_table_file)

    def diffuse_pheromones(self):

        new_wolf = self.wolf_pheromone_layer.data.copy()
        new_sheep = self.sheep_pheromone_layer.data.copy()

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                current_wolf = self.wolf_pheromone_layer.data[x, y]
                current_sheep = self.sheep_pheromone_layer.data[x, y]

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    nx = nx % self.grid.width
                    ny = ny % self.grid.height

                    new_wolf[nx, ny] += current_wolf * self.diffusion_rate
                    new_sheep[nx, ny] += current_sheep * self.diffusion_rate
                    #bilanciamento dei feromoni(manteniamo la stessa massa)
                    new_wolf[x, y] -= current_wolf * self.diffusion_rate
                    new_sheep[x, y] -= current_sheep * self.diffusion_rate

        self.wolf_pheromone_layer.data = new_wolf
        self.sheep_pheromone_layer.data = new_sheep

    def step(self):
        if self.count_agents(Sheep) == 0 or (self.get_steps() >= self.max_steps):
            self.save_q_tables()
            self.datacollector.collect(self)
            self.running = False
        else:
            self.agents.shuffle_do("step")
            self.datacollector.collect(self)

            if self.render_pheromone:
                for agent in self.agents:
                    if isinstance(agent, Pheromones):
                        agent.apply_diffusion()
            else:
                self.evaporate_pheromones()
                self.diffuse_pheromones()

    def __del__(self):
        self.save_q_tables()

    def get_steps(self):
        return self.steps
    def count_agents(self, agent_type):
        return sum(1 for agent in self.agents if isinstance(agent, agent_type))
    def evaporate_pheromones(self):
        if self.render_pheromone:
            return
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                self.wolf_pheromone_layer.set_cell((x, y), max(0, self.wolf_pheromone_layer.data[
                    x, y] - self.pheromone_evaporation))
                self.sheep_pheromone_layer.set_cell((x, y), max(0, self.sheep_pheromone_layer.data[
                    x, y] - self.pheromone_evaporation))






