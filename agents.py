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
    def __init__(self, actions=[0, 1, 2, 3], alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01,
                 q_table_file=None, q_learning=None):
        if q_learning:
            self.actions = q_learning.actions
            self.alpha = q_learning.alpha
            self.gamma = q_learning.gamma
            self.epsilon = q_learning.epsilon
            self.epsilon_decay = q_learning.epsilon_decay
            self.min_epsilon = q_learning.min_epsilon

        else:
            self.actions = actions
            self.alpha = alpha  # learning rate
            self.gamma = gamma  # discount factor
            self.epsilon = epsilon  # exploration rate iniziale
            self.epsilon_decay = epsilon_decay  # fattore di decadimento
            self.min_epsilon = min_epsilon  # valore minimo di exploration

        self.q_table = {}
        self.training = True
        #print(os.path.exists(q_table_file))
        if q_table_file and os.path.exists(q_table_file):
            self.load_q_table(q_table_file)

        #print("Creato q_learning con azioni: ", self.actions)

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

        #print(f"Salvato q_table in {filename} (dimensione: {len(serializable_q_table)} stati)")

    def load_q_table(self, filename):
        if not os.path.exists(filename):
            return
        else:
            with open(filename, 'r') as f:
                serializable_q_table = json.load(f)

            self.q_table = {
                eval(k): {int(ak): float(av) for ak, av in v.items()}
                for k, v in serializable_q_table.items()
            }

        #print(f"Caricato q_table da {filename} (dimensione: {len(self.q_table)} stati)")

    def get_state(self, wolf, pheromones, sheep_present):

        threshold = wolf.model.pheromone_treshold
        filtered_pheromones = []
        for ph in pheromones:
            wolf_conc = ph.wolf_concentration if ph.wolf_concentration >= threshold else 0
            sheep_conc = ph.sheep_concentration if ph.sheep_concentration >= threshold else 0
            filtered_pheromones.append(Pheromone(wolf_concentration=wolf_conc, sheep_concentration=sheep_conc))

        pheromones = filtered_pheromones


        differences = [ph.sheep_concentration - ph.wolf_concentration for ph in pheromones]
        max_index = int(np.argmax(differences))

        sheep_presence = int(sheep_present)

        closest_dist = wolf.model.get_closest_sheep_distance(wolf.pos)
        if closest_dist <= 1.5:
            dist_category = 0
        elif closest_dist <= 3.5:
            dist_category = 1
        elif closest_dist <= 5.5:
            dist_category = 2
        else:
            dist_category = 3

        return (max_index, sheep_presence, dist_category)
    def choose_action(self, state):

        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
            

        if self.training and random.random() < self.epsilon:

            #action_counts = {a: self.q_table[state][a] for a in self.actions}
            #least_tried = min(action_counts, key=action_counts.get)
            #return least_tried

            return random.choice(self.actions)
        else:
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}

        old_value = self.q_table[state][action]
        next_max = max(self.q_table.get(next_state, {a: 0 for a in self.actions}).values())

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

        self.q_table[state][action] = np.clip(new_value, -100, 100)

        #self.decay_epsilon() ######il decay lo faccio dopo ogni simulazione non dopo ogni step



class Animal(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.steps = 0
        self.movement_speed = 1

    def move(self):
        self.steps += 1
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False, radius=self.movement_speed
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

    def get_best_step(self, possible_steps, pheromones, is_wolf, action=None):

        threshold = self.model.pheromone_treshold
        filtered_pheromones = []
        for ph in pheromones:
            wolf_conc = ph.wolf_concentration if ph.wolf_concentration >= threshold else 0
            sheep_conc = ph.sheep_concentration if ph.sheep_concentration >= threshold else 0
            filtered_pheromones.append(Pheromone(wolf_concentration=wolf_conc, sheep_concentration=sheep_conc))
#
        pheromones = filtered_pheromones

        if is_wolf:
            if action == 0:
                pheromone_concentrations = [ph.sheep_concentration for ph in pheromones]
                max_pheromone = max(pheromone_concentrations)
                return [step for step, conc in zip(possible_steps, pheromone_concentrations) if conc == max_pheromone]
            elif action == 3:
                pheromone_concentrations = [ph.wolf_concentration for ph in pheromones]
                min_pheromone = min(pheromone_concentrations)
                return [step for step, conc in zip(possible_steps, pheromone_concentrations) if conc == min_pheromone]
            elif action == -1:
                pheromone_differences = [ph.sheep_concentration - ph.wolf_concentration for ph in pheromones]
                max_difference = max(pheromone_differences)
                return [step for step, diff in zip(possible_steps, pheromone_differences) if diff == max_difference]

        else:
            #return [self.random.choice(possible_steps)]
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
    def __init__(self, model, q_table_file="q_table_avg.json", q_learning=None):
            super().__init__(model)
            self.use_learning = self.model.learning
            self.q_learning = q_learning

            self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

            self.capture_intervals = []

            if self.use_learning:

                if self.model.testing:

                    self.q_learning = QLearning()
                    self.q_learning.load_q_table(model.q_table_file)
                    self.q_learning.epsilon = 0
                    self.q_learning.alpha = 0

                elif self.q_learning != None:

                    self.q_learning = QLearning(q_learning=self.model.q_learning,
                                                q_table_file=q_table_file)
                else:

                    self.q_learning = QLearning(
                        actions=[0, 1, 3],
                        alpha=0.2,
                        gamma=0.95,
                        epsilon=0.5,
                        epsilon_decay=0.995,
                        min_epsilon=0.005,
                        q_table_file=q_table_file
                    )


                self.rewards = []

            self.sheep_eaten = 0
            self.steps_since_last_capture = 0
            self.stayed = False
            self.eaten = False

            self.last_sheep_distance = float('inf')
            self.last_action = None
            self.current_action = None

    def __del__(self):
        if self.use_learning and hasattr(self, 'q_table_file') and self.q_table_file:
            self.q_learning.save_q_table(self.q_table_file)
            self.q_learning.save_q_table(self.q_table_file)

    def calculate_reward(self):

        base_reward = 0

        if self.eaten:

            base_reward += 10.0

            self.last_sheep_distance = self.model.get_closest_sheep_distance(self.pos)
            return base_reward

        current_dist = self.model.get_closest_sheep_distance(self.pos)

        if hasattr(self, 'last_sheep_distance') and self.steps > 0:
            dist_change = self.last_sheep_distance - current_dist

            distance_reward = dist_change# * 2
            base_reward += distance_reward

        self.last_sheep_distance = current_dist

        return base_reward
    def step(self):

        if not self.q_learning.actions == [0,1,2,3]:
            #print("rilascio")
            self.update_pheromone()

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

        sheep_present = any(any(isinstance(obj, Sheep) for obj in self.model.grid.get_cell_list_contents(step)) for step in possible_steps)

        if self.use_learning:

            state = self.q_learning.get_state(self, pheromones, sheep_present)
            action = self.q_learning.choose_action(state)
            self.current_action = action

            self.action_counts[action] += 1

        else:
            action = -1

        if action == 0 or action == 3:
            best_steps = self.get_best_step(possible_steps, pheromones, True, action)
            if best_steps:
                self.model.grid.move_agent(self, self.random.choice(best_steps))
        elif action == 1:
            if possible_steps:
                self.model.grid.move_agent(self, self.random.choice(possible_steps))
        elif action == 2:
            self.update_pheromone()
        else:
            best_steps = self.get_best_step(possible_steps, pheromones, True, action)
            if best_steps:
                self.model.grid.move_agent(self, self.random.choice(best_steps))

        if not self.q_learning.actions == [0, 1, 2, 3]:
            self.update_pheromone()

        self.eaten = self.eat_sheep()

        if self.use_learning:

            reward = self.calculate_reward()

            if self.eaten:

                if self.steps_since_last_capture > 0:
                    self.capture_intervals.append(self.steps_since_last_capture)

                self.steps_since_last_capture = 0
                self.eaten = False

            else:
                self.steps_since_last_capture += 1

            self.rewards.append(reward)

            next_possible_steps = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False
            )
            next_pheromones = [
                Pheromone(
                    wolf_concentration=self.model.wolf_pheromone_layer.data[x, y],
                    sheep_concentration=self.model.sheep_pheromone_layer.data[x, y]
                ) for (x, y) in next_possible_steps
            ] if not self.model.render_pheromone else [
                next((obj.pheromone for obj in self.model.grid.get_cell_list_contents(step) if
                      isinstance(obj, Pheromones)),
                     Pheromone())
                for step in next_possible_steps
            ]

            next_sheep_present = any(
                any(isinstance(obj, Sheep) for obj in self.model.grid.get_cell_list_contents(step)) for step in
                next_possible_steps)

            next_state = self.q_learning.get_state(self, next_pheromones, next_sheep_present)
            self.q_learning.learn(state, action, reward, next_state)

        self.steps += 1

    def eat_sheep(self):
        sheep = [obj for obj in self.model.grid.get_cell_list_contents([self.pos]) if isinstance(obj, Sheep)]
        if sheep:
            sheep_to_eat = self.random.choice(sheep)
            sheep_to_eat.alive = False
            self.sheep_eaten += 1
            if self.model.respawn:
                sheep_to_eat.respawn()
            else:
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
        self.movement_speed = 2

    def respawn(self):
        self.model.grid.move_agent(self, self.random_position())
        self.alive = True

    def random_position(self):
        return (
            self.random.randrange(self.model.grid.width),
            self.random.randrange(self.model.grid.height)
        )
    def step(self):
        if self.alive:
            self.update_pheromone()
            super().step()
            self.update_pheromone()


class Pheromones(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.pheromone = Pheromone()
        self.next_wolf = 0.0
        self.next_sheep = 0.0

    def prepare_diffusion(self):



        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]


        fraction_per_direction = self.model.diffusion_rate / len(directions)

        for dx, dy in directions:
            nx, ny = (self.pos[0] + dx) % self.model.grid.width, (self.pos[1] + dy) % self.model.grid.height
            neighbors = self.model.grid.get_cell_list_contents((nx, ny))
            for neighbor in neighbors:
                if isinstance(neighbor, Pheromones):

                    wolf_diffused = self.pheromone.wolf_concentration * fraction_per_direction
                    sheep_diffused = self.pheromone.sheep_concentration * fraction_per_direction

                    neighbor.next_wolf += wolf_diffused
                    neighbor.next_sheep += sheep_diffused

                    self.next_wolf -= wolf_diffused
                    self.next_sheep -= sheep_diffused

    def apply_diffusion(self):
        #print("faccio diffusione, ", self.next_sheep)
        self.pheromone.wolf_concentration += self.next_wolf
        self.pheromone.sheep_concentration += self.next_sheep
        self.next_wolf = 0.0
        self.next_sheep = 0.0

    def step(self):
        self.pheromone.sheep_concentration *= (1 - self.model.pheromone_evaporation)
        self.pheromone.wolf_concentration *= (1 - self.model.pheromone_evaporation)
        self.prepare_diffusion()