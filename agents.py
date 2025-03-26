import random
from dataclasses import dataclass
import numpy as np
from mesa import Agent
import model
import json
import os

@dataclass
class Pheromone:
    wolf_concentration: float = 0.0
    sheep_concentration: float = 0.0


class QLearning:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01,
                 q_table_file=None):
        self.actions = actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate iniziale
        self.epsilon_decay = epsilon_decay  # fattore di decadimento
        self.min_epsilon = min_epsilon  # valore minimo di exploration
        self.q_table = {}
        self.training = True

        if q_table_file and os.path.exists(q_table_file):
            self.load_q_table(q_table_file)

    def decay_epsilon(self):
        if self.training:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename):
        serializable_q_table = {
            str(k): {int(ak): float(av) for ak, av in v.items()}
            for k, v in self.q_table.items()
        }

        with open(filename, 'w') as f:
            json.dump(serializable_q_table, f, indent=2)

    def load_q_table(self, filename):

        with open(filename, 'r') as f:
            serializable_q_table = json.load(f)

        self.q_table = {
            eval(k): {int(ak): float(av) for ak, av in v.items()}
            for k, v in serializable_q_table.items()
        }

    def get_state(self, wolf, pheromones):

        if not pheromones:
            return (0, 0, int(wolf.stayed))

        sheep_vals = [ph.sheep_concentration for ph in pheromones]
        wolf_vals = [ph.wolf_concentration for ph in pheromones]

        sheep_level = min(2, int(np.mean(sheep_vals) * 3))
        wolf_level = min(2, int(np.mean(wolf_vals) * 3))

        return (sheep_level, wolf_level, int(wolf.stayed))

    def choose_action(self, state):

        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}

        if self.training and random.random() < self.epsilon:
            exp_values = np.exp([self.q_table[state][a] for a in self.actions])
            probs = exp_values / np.sum(exp_values)
            return np.random.choice(self.actions, p=probs)
        else:
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}

        old_value = self.q_table[state][action]
        next_max = max(self.q_table.get(next_state, {a: 0 for a in self.actions}).values())

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

        self.q_table[state][action] = np.clip(new_value, -10, 10)

        self.decay_epsilon()



class Animal(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.steps = 0
        self.sheep_eaten = 0

    def move(self):
        self.steps += 1
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )

        pheromones = [
            Pheromone(
                wolf_concentration=self.model.wolf_pheromone_layer.data[x, y],
                sheep_concentration=self.model.sheep_pheromone_layer.data[x, y]
            ) for (x, y) in possible_steps
        ] if not self.model.render_pheromone else [
            next((obj.pheromone for obj in self.model.grid.get_cell_list_contents(step) if isinstance(obj, Pheromones)),
                 Pheromone())
            for step in possible_steps
        ]

        best_steps = self.get_best_step(possible_steps, pheromones, isinstance(self, Wolf))
        self.model.grid.move_agent(self, self.random.choice(best_steps))

    def get_best_step(self, possible_steps, pheromones, is_wolf):
        if is_wolf:
            pheromone_differences = [ph.sheep_concentration - ph.wolf_concentration for ph in pheromones]
            max_difference = max(pheromone_differences)
            return [step for step, diff in zip(possible_steps, pheromone_differences) if diff == max_difference]
        else:
            pheromone_concentrations = [ph.wolf_concentration for ph in pheromones]
            min_pheromone = min(pheromone_concentrations)
            return [step for step, conc in zip(possible_steps, pheromone_concentrations) if conc == min_pheromone]

    def step(self):
        self.move()

    def update_pheromone(self):
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        if self.model.render_pheromone:
            ph = next((obj for obj in cell_contents if isinstance(obj, Pheromones)), None)
            if ph:
                if isinstance(self, Wolf):
                    ph.pheromone.wolf_concentration += self.model.pheromone_added
                else:
                    ph.pheromone.sheep_concentration += self.model.pheromone_added
        else:
            if isinstance(self, Wolf):
                self.model.wolf_pheromone_layer.set_cell(
                    self.pos, self.model.wolf_pheromone_layer.data[self.pos] + self.model.pheromone_added
                )
            else:
                self.model.sheep_pheromone_layer.set_cell(
                    self.pos, self.model.sheep_pheromone_layer.data[self.pos] + self.model.pheromone_added
                )


class Wolf(Animal):
    def __init__(self, model, q_table_file="q_table.json"):
        super().__init__(model)
        self.q_learning = QLearning(
            actions=[0, 1, 2],
            alpha=0.2,
            gamma=0.95,
            epsilon=0.3,
            epsilon_decay=0.998,
            min_epsilon=0.05,
            q_table_file=q_table_file
        )
        self.last_capture_step = 0
        self.steps_since_last_capture = 0
        self.fixed_reward = 10
        self.step_penalty = -0.05
        self.distance_penalty = 0.1
        self.stayed = False

    def __del__(self):
        if hasattr(self, 'q_table_file') and self.q_table_file:
            self.q_learning.save_q_table(self.q_table_file)

    def calculate_reward(self, sheep_eaten, steps_since_last):
        if sheep_eaten:
            time_bonus = 1 / (steps_since_last + 1)
            return self.fixed_reward + self.distance_penalty * time_bonus
        return self.step_penalty

    def step(self):

        if(self.stayed):
            possible_steps = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False, radius=2
            )
            self.stayed = False
        else:
            possible_steps = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False, radius=1
            )

        pheromones = [
            Pheromone(
                wolf_concentration=self.model.wolf_pheromone_layer.data[x, y],
                sheep_concentration=self.model.sheep_pheromone_layer.data[x, y]
            ) for (x, y) in possible_steps
        ] if not self.model.render_pheromone else [
            next((obj.pheromone for obj in self.model.grid.get_cell_list_contents(step) if isinstance(obj, Pheromones)),
                 Pheromone())
            for step in possible_steps
        ]


        state = self.q_learning.get_state(self, pheromones)

        action = self.q_learning.choose_action(state)

        if action == 0:
            best_steps = self.get_best_step(possible_steps, pheromones, True)
            if best_steps:
                self.model.grid.move_agent(self, self.random.choice(best_steps))
        elif action == 1:
            if possible_steps:
                self.model.grid.move_agent(self, self.random.choice(possible_steps))
        elif action == 2:
            self.stayed = True

        reward = self.calculate_reward(False, self.steps_since_last_capture)

        sheep_eaten = self.eat_sheep()
        if sheep_eaten:
            reward = self.calculate_reward(True, self.model.get_steps() - self.last_capture_step)
            self.last_capture_step = self.model.get_steps()
            self.steps_since_last_capture = 0
        else:
            self.steps_since_last_capture += 1

        self.update_pheromone()

        next_possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        next_pheromones = [
            Pheromone(
                wolf_concentration=self.model.wolf_pheromone_layer.data[x, y],
                sheep_concentration=self.model.sheep_pheromone_layer.data[x, y]
            ) for (x, y) in next_possible_steps
        ] if not self.model.render_pheromone else [
            next((obj.pheromone for obj in self.model.grid.get_cell_list_contents(step) if isinstance(obj, Pheromones)),
                 Pheromone())
            for step in next_possible_steps
        ]
        next_state = self.q_learning.get_state(self, next_pheromones)

        self.q_learning.learn(state, action, reward, next_state)

        self.steps += 1

    def eat_sheep(self):
        sheep = [obj for obj in self.model.grid.get_cell_list_contents([self.pos]) if isinstance(obj, Sheep)]
        if sheep:
            sheep_to_eat = self.random.choice(sheep)
            sheep_to_eat.alive = False
            self.sheep_eaten += 1
            self.model.grid.remove_agent(sheep_to_eat)
            self.model.agents.remove(sheep_to_eat)
            return True
        return False

    def get_sheep_eaten(self):
        return self.sheep_eaten if isinstance(self, Wolf) else None

    def avg_step_per_sheep(self):
        return self.steps / self.sheep_eaten if isinstance(self, Wolf) and self.sheep_eaten > 0 else None


class Sheep(Animal):
    def __init__(self, model):
        super().__init__(model)
        self.alive = True

    def step(self):
        if self.alive:
            super().step()
            self.update_pheromone()


class Pheromones(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.pheromone = Pheromone()

    def step(self):
        self.pheromone.sheep_concentration = max(0,
                                                 self.pheromone.sheep_concentration - self.model.pheromone_evaporation)
        self.pheromone.wolf_concentration = max(0, self.pheromone.wolf_concentration - self.model.pheromone_evaporation)