import mesa
import json
from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
from agents import Wolf, Sheep
import numpy as np


with open("config.json", "r") as f:
    config = json.load(f)


class WolfSheepModel(Model):

    def __init__(self, width=config["grid_width"], height=config["grid_height"], initial_wolves=config["num_wolves"],
                 initial_sheep=config["num_sheep"], pheromone_evaporation=config["pheromone_evaporation"],
                 pheromone_added=config["pheromone_added"], seed=None):

        super().__init__(seed=seed)
        self.pheromone_evaporation = pheromone_evaporation
        self.pheromone_added = pheromone_added
        self.grid = MultiGrid(width, height, torus=True)
        self.running = True
        self.wolf_pheromone_layer = PropertyLayer("wolf_pheromone", width, height, 0.0)
        self.sheep_pheromone_layer = PropertyLayer("sheep_pheromone", width, height, 0.0)
        self.datacollector = DataCollector(
            model_reporters={
                "Steps": self.get_steps,
            },
            agent_reporters={
                "Sheep_eaten": Wolf.get_sheep_eaten,
                "Avg_step_per_sheep": Wolf.avg_step_per_sheep
            }
        )

        wolves = Wolf.create_agents(model=self, n=initial_wolves)

        # Create x and y positions for agents
        x = self.rng.integers(0, self.grid.width, size=(initial_wolves,))
        y = self.rng.integers(0, self.grid.height, size=(initial_wolves,))
        for a, i, j in zip(wolves, x, y):
            # Add the agent to a random grid cell
            self.grid.place_agent(a, (i, j))

        sheep = Sheep.create_agents(model=self, n=initial_sheep)

        # Create x and y positions for agents
        x = self.rng.integers(0, self.grid.width, size=(initial_sheep,))
        y = self.rng.integers(0, self.grid.height, size=(initial_sheep,))
        for a, i, j in zip(sheep, x, y):

            self.grid.place_agent(a, (i, j))


    def get_steps(self):
        return self.steps
    def step(self):
        if self.count_sheep() == 0:
            self.datacollector.collect(self)
            self.running = False

        else:
            self.update_wolf_pheromone()
            self.update_sheep_pheromone()
            self.agents.shuffle_do("step")
            self.datacollector.collect(self)

    def update_wolf_pheromone(self):

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                current_pheromone = self.wolf_pheromone_layer.data[x, y]
                new_pheromone = max(0.0, current_pheromone - self.pheromone_evaporation)  # Evaporazione
                self.wolf_pheromone_layer.set_cell((x, y), new_pheromone)

        for agent in self.agents:
            if isinstance(agent, Wolf) and agent.pos is not None:
                (x, y) = agent.pos
                current_pheromone = self.wolf_pheromone_layer.data[x, y]
                new_pheromone = current_pheromone + self.pheromone_added
                self.wolf_pheromone_layer.set_cell((x, y), new_pheromone)

    def update_sheep_pheromone(self):

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                current_pheromone = self.sheep_pheromone_layer.data[x, y]
                new_pheromone = max(0.0, current_pheromone - self.pheromone_evaporation)  # Evaporazione
                self.sheep_pheromone_layer.set_cell((x, y), new_pheromone)

        for agent in self.agents:
            if isinstance(agent, Sheep) and agent.pos is not None:
                (x, y) = agent.pos
                current_pheromone = self.sheep_pheromone_layer.data[x, y]
                new_pheromone = current_pheromone + self.pheromone_added
                self.sheep_pheromone_layer.set_cell((x, y), new_pheromone)

    def count_wolves(self):
        count = sum(1 for agent in self.agents if isinstance(agent, Wolf))
        return count
    def count_sheep(self):
        count = sum(1 for agent in self.agents if isinstance(agent, Sheep))
        return count

    def get_pheromone_map(self, pheromone_type="wolf"):
        if pheromone_type == "wolf":
            return self.wolf_pheromone_layer.data
        elif pheromone_type == "sheep":
            return self.sheep_pheromone_layer.data
        return np.zeros((self.grid.width, self.grid.height))






