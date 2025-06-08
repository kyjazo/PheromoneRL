import math

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
                 pheromone_added=config["pheromone_added"], render_pheromone=False,  q_table_file="q_table_avg.json", max_steps=200,
                 diffusion_rate=config["diffusion_rate"], respawn=True, learning=True, q_learning=None, testing=False, seed=None):
         super().__init__(seed=seed)

         self.testing = testing
         self.q_learning = q_learning
         self.respawn = respawn
         self.learning = learning
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
                 "Sheep_Alive": self.get_sheeps
             },
             agent_reporters={
                 "Sheep_eaten": lambda a: int(a.sheep_eaten) if hasattr(a, 'sheep_eaten') else None,
                  "Reward": lambda a: float(np.mean(a.rewards))  # Usa la media delle reward
                  if hasattr(a, 'rewards') and a.rewards else None,
                 "Action_0": lambda a: a.action_counts[0] if hasattr(a, "action_counts") else None,
                 "Action_1": lambda a: a.action_counts[1] if hasattr(a, "action_counts") else None,
                 #"Action_2": lambda a: a.action_counts[2] if hasattr(a, "action_counts") else None,
                 "Action_3": lambda a: a.action_counts[3] if hasattr(a, "action_counts") else None,
                 "Capture_Intervals": lambda a: a.capture_intervals if hasattr(a, 'capture_intervals') else [],


             }
         )




         self.place_agents(Wolf, initial_wolves)
         self.place_agents(Sheep, initial_sheep)
         if render_pheromone:
             pheromones = Pheromones.create_agents(model=self, n=width * height)
             positions = np.array(np.meshgrid(np.arange(width), np.arange(height))).T.reshape(-1, 2)
             for agent, pos in zip(pheromones, positions):
                 self.grid.place_agent(agent, tuple(pos))

    #def place_agents(self, agent_class, num_agents):
#
    #    agents = agent_class.create_agents(model=self, n=num_agents)
    #    positions = np.column_stack((self.rng.integers(0, self.grid.width, size=num_agents),
    #                                 self.rng.integers(0, self.grid.height, size=num_agents)))
    #    for agent, pos in zip(agents, positions):
    #        self.grid.place_agent(agent, tuple(pos))

    def place_agents(self, agent_class, num_agents):

        if agent_class.__name__ == "Wolf":
            agents = agent_class.create_agents(model=self, n=num_agents, q_table_file=self.q_table_file)

            center_x, center_y = self.grid.width // 2, self.grid.height // 2
            positions = [(center_x + dx, center_y + dy)
                         for dx in range(-2, 3) for dy in range(-2, 3)]
            self.random.shuffle(positions)
            positions = positions[:num_agents]

        elif agent_class.__name__ == "Sheep":
            agents = agent_class.create_agents(model=self, n=num_agents)
            positions = []
            while len(positions) < num_agents:
                x = self.random.randint(0, self.grid.width - 1)
                y = self.random.choice([0, self.grid.height - 1])
                if (x, y) not in positions:
                    positions.append((x, y))
                if len(positions) < num_agents:
                    y = self.random.randint(0, self.grid.height - 1)
                    x = self.random.choice([0, self.grid.width - 1])
                    if (x, y) not in positions:
                        positions.append((x, y))
            positions = positions[:num_agents]

        else:
            agents = agent_class.create_agents(model=self, n=num_agents)
            positions = np.column_stack((self.rng.integers(0, self.grid.width, size=num_agents),
                                         self.rng.integers(0, self.grid.height, size=num_agents)))

        for agent, pos in zip(agents, positions):
            self.grid.place_agent(agent, tuple(pos))



    def get_closest_sheep_distance(self, pos, radius=6):

        neighbors = self.grid.get_neighbors(pos, moore=True, include_center=False, radius=radius)
        sheep = [agent for agent in neighbors if isinstance(agent, Sheep) and agent.alive]

        if not sheep:
            return radius + 1

        return min(self.get_distance(pos, s.pos) for s in sheep)

    def get_distance(self, pos1, pos2):

        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        dx = min(dx, self.grid.width - dx)
        dy = min(dy, self.grid.height - dy)

        return math.sqrt(dx ** 2 + dy ** 2)

    def save_q_tables(self):
        if not self.learning or not hasattr(self, 'q_table_file') or not self.q_table_file:
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
            temp_q_learning = QLearning(actions=[0, 1, 3]) #eliminata azione 2
            temp_q_learning.q_table = q_tables
            temp_q_learning.save_q_table(self.q_table_file)
            #print("Saved: ", self.q_table_file)




    def diffuse_pheromones(self):

        new_wolf = self.wolf_pheromone_layer.data.copy()
        new_sheep = self.sheep_pheromone_layer.data.copy()


        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]

        fraction_per_direction = self.diffusion_rate / len(directions)

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                current_wolf = self.wolf_pheromone_layer.data[x, y]
                current_sheep = self.sheep_pheromone_layer.data[x, y]

                total_diffused_wolf = 0
                total_diffused_sheep = 0

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    nx = nx % self.grid.width
                    ny = ny % self.grid.height

                    wolf_diffused = current_wolf * fraction_per_direction
                    sheep_diffused = current_sheep * fraction_per_direction

                    new_wolf[nx, ny] += wolf_diffused
                    new_sheep[nx, ny] += sheep_diffused
                    total_diffused_wolf += wolf_diffused
                    total_diffused_sheep += sheep_diffused

                new_wolf[x, y] -= total_diffused_wolf
                new_sheep[x, y] -= total_diffused_sheep

        self.wolf_pheromone_layer.data = new_wolf
        self.sheep_pheromone_layer.data = new_sheep

    def decay_epsilon(self):
        if self.learning:
            self.q_learning.epsilon = max(self.q_learning.min_epsilon, self.q_learning.epsilon * self.q_learning.epsilon_decay)
    def step(self):

        if self.count_agents(Sheep) == 0 or (self.get_steps() >= self.max_steps):
            if not self.testing:
                self.decay_epsilon()
            #print("finita simulazione, chiamo save table")
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

            #self.save_q_tables()

    def __del__(self):
        #print("Chiamo save tables")
        self.save_q_tables()

    def get_steps(self):
        return self.steps
    def get_sheeps(self):
        return sum(1 for agent in self.agents
                   if isinstance(agent, Sheep) and agent.alive)
    def count_agents(self, agent_type):
        return sum(1 for agent in self.agents if isinstance(agent, agent_type))

    def evaporate_pheromones(self):
        if self.render_pheromone:
            return
        for x in range(self.grid.width):
            for y in range(self.grid.height):

                current_wolf = self.wolf_pheromone_layer.data[x, y]
                current_sheep = self.sheep_pheromone_layer.data[x, y]

                self.wolf_pheromone_layer.set_cell((x, y), current_wolf * (1 - self.pheromone_evaporation))
                self.sheep_pheromone_layer.set_cell((x, y), current_sheep * (1 - self.pheromone_evaporation))






