import mesa
from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
from agents import Wolf, Sheep
import numpy as np

class WolfSheepModel(Model):
    """
    Modello per la simulazione di lupi e pecore con feromoni.
    """
    def __init__(self, width=20, height=20, initial_wolves=5, initial_sheep=20, seed=None):
        super().__init__(seed=seed)

        self.grid = MultiGrid(width, height, torus=True)
        self.running = True
        self.wolf_pheromone_layer = PropertyLayer("wolf_pheromone", width, height, 0.0)
        self.sheep_pheromone_layer = PropertyLayer("sheep_pheromone", width, height, 0.0)
        self.datacollector = DataCollector(
            model_reporters={
                "Wolves": self.count_wolves,
                "Sheep": self.count_sheep
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
                new_pheromone = max(0.0, current_pheromone - 0.1)  # Evaporazione
                self.wolf_pheromone_layer.set_cell((x, y), new_pheromone)

        for agent in self.agents:
            if isinstance(agent, Wolf) and agent.pos is not None:
                (x, y) = agent.pos
                current_pheromone = self.wolf_pheromone_layer.data[x, y]
                new_pheromone = current_pheromone + 0.5
                self.wolf_pheromone_layer.set_cell((x, y), new_pheromone)
#rendere il modello parametrizzato, iniziare a pensare ad un setup possibile per il rl
#tenere traccia di metriche, oltre a risorse impiegate,
#pettingzoo
#spazio di osservazione(cosa vede), spazio di azione(cosa fa), reward
#mesa offre ottimizzazioni performance?
#ottimizzazioni generali, es: array.np piuttosto che lista python
#visualizzare i feromoni
    def update_sheep_pheromone(self):

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                current_pheromone = self.sheep_pheromone_layer.data[x, y]
                new_pheromone = max(0.0, current_pheromone - 0.1)  # Evaporazione
                self.sheep_pheromone_layer.set_cell((x, y), new_pheromone)

        for agent in self.agents:
            if isinstance(agent, Sheep) and agent.pos is not None:
                (x, y) = agent.pos
                current_pheromone = self.sheep_pheromone_layer.data[x, y]
                new_pheromone = current_pheromone + 0.5
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






