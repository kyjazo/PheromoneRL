from mesa import Agent
import mesa
import numpy as np

import model


class Animal(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.steps = 0
        self.sheep_eaten = 0

    def move(self):
        self.steps += 1
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )

        pheromones = [self.model.grid.get_cell_list_contents(step) for step in possible_steps]
        pheromones = [[obj for obj in cell if isinstance(obj, Pheromone)][0] for cell in pheromones]

        if isinstance(self, Wolf):
            pheromone_differences = [
                pheromone.sheep_concentration - pheromone.wolf_concentration
                for pheromone in pheromones
            ]
            max_difference = max(pheromone_differences)
            best_steps = [
                step for step, diff in zip(possible_steps, pheromone_differences)
                if diff == max_difference
            ]
        else:
            pheromone_concentrations = [pheromone.wolf_concentration for pheromone in pheromones]
            min_pheromone = min(pheromone_concentrations)
            best_steps = [
                step for step, conc in zip(possible_steps, pheromone_concentrations)
                if conc == min_pheromone
            ]

        chosen_step = self.random.choice(best_steps)
        self.model.grid.move_agent(self, chosen_step)


    def step(self):
        self.move()


class Wolf(Animal):
    def __init__(self, model):
        super().__init__(model)

    def step(self):
        self.move()
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        sheep = [obj for obj in cell_contents if isinstance(obj, Sheep)]
        if sheep:
            sheep_to_eat = self.random.choice(sheep)
            sheep_to_eat.alive = False
            self.sheep_eaten += 1
            self.model.grid.remove_agent(sheep_to_eat)
            self.model.agents.remove(sheep_to_eat)

        pheromone = next(obj for obj in cell_contents if isinstance(obj, Pheromone))
        pheromone.wolf_concentration += self.model.pheromone_added

    def get_sheep_eaten(self):
        if isinstance(self, Wolf):
            return self.sheep_eaten

    def avg_step_per_sheep(self):
        if isinstance(self, Wolf):
            return self.steps / self.sheep_eaten if self.sheep_eaten > 0 else 0


class Sheep(Animal):
    def __init__(self, model):
        super().__init__(model)
        self.alive = True

    def step(self):
        if self.alive:
            self.move()
            cell_contents = self.model.grid.get_cell_list_contents([self.pos])
            pheromone = next(obj for obj in cell_contents if isinstance(obj, Pheromone))
            pheromone.sheep_concentration += self.model.pheromone_added


class Pheromone(Agent):
    def __init__(self, model):
        super().__init__(model)

        self.sheep_concentration = 0.0
        self.wolf_concentration = 0.0

    def step(self):

        self.sheep_concentration = max(0, self.sheep_concentration - self.model.pheromone_evaporation)
        self.wolf_concentration = max(0, self.wolf_concentration - self.model.pheromone_evaporation)

