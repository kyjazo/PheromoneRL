from dataclasses import dataclass

import numpy
from mesa import Agent
import mesa
import numpy as np

import model

@dataclass
class Pheromone:
    wolf_concentration: numpy.float64 = 0.0
    sheep_concentration: numpy.float64 = 0.0




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

        if(self.model.render_pheromone):
            pheromones = [self.model.grid.get_cell_list_contents(step) for step in possible_steps]
            pheromones = [[obj for obj in cell if isinstance(obj, Pheromones)][0] for cell in pheromones]
        else:

            pheromones = [
                Pheromone(
                    wolf_concentration=self.model.wolf_pheromone_layer.data[x, y],
                    sheep_concentration=self.model.sheep_pheromone_layer.data[x, y]
                )
                for (x, y) in possible_steps
            ]

        if isinstance(self, Wolf):
            if(self.model.render_pheromone):
                pheromone_differences = [
                    ph.pheromone.sheep_concentration - ph.pheromone.wolf_concentration
                    for ph in pheromones
                ]
                max_difference = max(pheromone_differences)
                best_steps = [
                    step for step, diff in zip(possible_steps, pheromone_differences)
                    if diff == max_difference
                ]
            else:
                pheromone_differences = [
                    ph.sheep_concentration - ph.wolf_concentration
                    for ph in pheromones
                ]
                max_difference = max(pheromone_differences)
                best_steps = [
                    step for step, diff in zip(possible_steps, pheromone_differences)
                    if diff == max_difference
                ]
        else:
            if(self.model.render_pheromone):
                pheromone_concentrations = [ph.pheromone.wolf_concentration for ph in pheromones]
                min_pheromone = min(pheromone_concentrations)
                best_steps = [
                    step for step, conc in zip(possible_steps, pheromone_concentrations)
                    if conc == min_pheromone
                ]
            else:
                pheromone_concentrations = [ph.wolf_concentration for ph in pheromones]
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

        if(self.model.render_pheromone):
            ph = next(obj for obj in cell_contents if isinstance(obj, Pheromones))
            ph.pheromone.wolf_concentration += self.model.pheromone_added
        else:
            new_pheromone = self.model.wolf_pheromone_layer.data[self.pos] + self.model.pheromone_added
            self.model.wolf_pheromone_layer.set_cell(self.pos, new_pheromone)

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
            if(self.model.render_pheromone):
                ph = next(obj for obj in cell_contents if isinstance(obj, Pheromones))
                ph.pheromone.sheep_concentration += self.model.pheromone_added
            else:
                new_pheromone = self.model.sheep_pheromone_layer.data[self.pos] + self.model.pheromone_added
                self.model.sheep_pheromone_layer.set_cell(self.pos, new_pheromone)


class Pheromones(Agent):
    def __init__(self, model):
        super().__init__(model)

        self.pheromone = Pheromone()


    def step(self):

        self.pheromone.sheep_concentration = max(0, self.pheromone.sheep_concentration - self.model.pheromone_evaporation)
        self.pheromone.wolf_concentration = max(0, self.pheromone.wolf_concentration - self.model.pheromone_evaporation)

