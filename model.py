import mesa
import json
from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
from agents import Wolf, Sheep, Pheromones
import numpy as np


with open("config.json", "r") as f:
    config = json.load(f)


class WolfSheepModel(Model):

    def __init__(self, width=config["grid_width"], height=config["grid_height"], initial_wolves=config["num_wolves"],
                 initial_sheep=config["num_sheep"], pheromone_evaporation=config["pheromone_evaporation"],
                 pheromone_added=config["pheromone_added"], render_pheromone=False, seed=None):

        super().__init__(seed=seed)
        self.render_pheromone = render_pheromone
        self.wolf_pheromone_layer = PropertyLayer("wolf_pheromone_layer", height=height, width=width, default_value=0.0)
        self.sheep_pheromone_layer = PropertyLayer("sheep_pheromone_layer", height=height, width=width, default_value=0.0)
        self.pheromone_evaporation = pheromone_evaporation
        self.pheromone_added = pheromone_added
        self.grid = MultiGrid(width, height, torus=True)
        self.running = True
        self.datacollector = DataCollector(
            model_reporters={
                "Steps": self.get_steps,
            },
            agent_reporters={
                "Sheep_eaten": Wolf.get_sheep_eaten,
                "Avg_step_per_sheep": Wolf.avg_step_per_sheep
            }
        )

        # Create wolves
        wolves = Wolf.create_agents(model=self, n=initial_wolves)
        positions = np.column_stack((self.rng.integers(0, self.grid.width, size=initial_wolves),
                                     self.rng.integers(0, self.grid.height, size=initial_wolves)))
        for agent, pos in zip(wolves, positions):
            self.grid.place_agent(agent, tuple(pos))

        # Create sheep
        sheep = Sheep.create_agents(model=self, n=initial_sheep)
        positions = np.column_stack((self.rng.integers(0, self.grid.width, size=initial_sheep),
                                     self.rng.integers(0, self.grid.height, size=initial_sheep)))
        for agent, pos in zip(sheep, positions):
            self.grid.place_agent(agent, tuple(pos))

        if(render_pheromone):
            # Create pheromones
            pheromones = Pheromones.create_agents(model=self, n=width * height)
            positions = np.array(np.meshgrid(np.arange(width), np.arange(height))).T.reshape(-1, 2)
            for agent, pos in zip(pheromones, positions):
                self.grid.place_agent(agent, tuple(pos))



    def get_steps(self):
        return self.steps


    def step(self):
        sheep_count = self.count_sheep()
        if sheep_count == 0:
            self.datacollector.collect(self)
            self.running = False
        else:
            self.agents.shuffle_do("step")
            self.datacollector.collect(self)
            if(not self.render_pheromone):
                self.evaporate_pheromones()

    def count_wolves(self):
        count = sum(1 for agent in self.agents if isinstance(agent, Wolf))
        return count
    def count_sheep(self):
        count = sum(1 for agent in self.agents if isinstance(agent, Sheep))
        return count

    def evaporate_pheromones(self):
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                pos=(x,y)
                wolf_value = self.wolf_pheromone_layer.data[x, y]
                self.wolf_pheromone_layer.set_cell(pos, max(0, wolf_value - self.pheromone_evaporation))

                sheep_value = self.sheep_pheromone_layer.data[x, y]
                self.sheep_pheromone_layer.set_cell(pos, max(0, sheep_value - self.pheromone_evaporation))







