from mesa import Agent
import mesa
import numpy as np
class Animal(Agent):
    def __init__(self, model):
        super().__init__(model)


    def move(self):

        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )

        if isinstance(self, Wolf):
            pheromone_concentrations_sheep = []
            pheromone_concentrations_wolf = []
            for step in possible_steps:
                pheromone_sheep = self.model.sheep_pheromone_layer.data[step]
                pheromone_concentrations_sheep.append(pheromone_sheep)

                pheromone_wolf = self.model.wolf_pheromone_layer.data[step]
                pheromone_concentrations_wolf.append(pheromone_wolf)

            pheromone_differences = [
                pheromone_sheep - pheromone_wolf
                for pheromone_sheep, pheromone_wolf in
                zip(pheromone_concentrations_sheep, pheromone_concentrations_wolf)
            ]

            max_difference = max(pheromone_differences)
            best_steps = [
                step for step, diff in zip(possible_steps, pheromone_differences)
                if diff == max_difference
            ]
        else:
            pheromone_concentrations = []
            for step in possible_steps:
                pheromone = self.model.wolf_pheromone_layer.data[step]
                pheromone_concentrations.append(pheromone)

            min_pheromone = min(pheromone_concentrations)
            best_steps = [
                step for step, conc in zip(possible_steps, pheromone_concentrations)
                if conc == min_pheromone
            ]

        # Se ci sono più direzioni con la stessa concentrazione massima, scegli una casualmente
        chosen_step = self.random.choice(best_steps)

        self.model.grid.move_agent(self, chosen_step)

    def step(self):
        """
        Definisce il comportamento ad ogni passo dell'animale.
        """
        self.move()


class Wolf(Animal):
    """
    Classe per i lupi che eredita da Animal.
    """
    def __init__(self, model,):
        super().__init__(model)
        self.q_table = {}  # Dizionario per memorizzare la Q-table
        self.learning_rate = 0.1  # Tasso di apprendimento
        self.discount_factor = 0.9  # Fattore di sconto
        self.epsilon = 1.0  # Probabilità di esplorazione (epsilon-greedy)
        self.epsilon_decay = 0.995  # Decadimento di epsilon
        self.epsilon_min = 0.01  # Valore minimo di epsilon

    def get_state(self):
        x, y = self.pos
        state = (x, y)
        return state

    def get_possible_actions(self):
        return self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )

    def get_reward(self):

        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        sheep = [obj for obj in cell_contents if isinstance(obj, Sheep)]
        if sheep:
            return 10
        else:
            return -1

    def update_q_table(self, state, action, reward, next_state):

        possible_actions = self.get_possible_actions()

        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(possible_actions))

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(possible_actions))

        if action not in possible_actions:
            return

        action_index = possible_actions.index(action)

        current_q = self.q_table[state][action_index]
        max_future_q = np.max(self.q_table[next_state])

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state][action_index] = new_q

    def choose_action(self, state):

        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.get_possible_actions()))

        possible_actions = self.get_possible_actions()

        if np.random.rand() < self.epsilon:

            return self.random.choice(possible_actions)
        else:

            return possible_actions[np.argmax(self.q_table[state])]

    def step(self):
        #state = self.get_state()
        #action = self.choose_action(state)
        #self.model.grid.move_agent(self, action)
        #reward = self.get_reward()
        #next_state = self.get_state()
        #self.update_q_table(state, action, reward, next_state)
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.move()

        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        sheep = [obj for obj in cell_contents if isinstance(obj, Sheep)]
        if sheep:
            sheep_to_eat = self.random.choice(sheep)
            sheep_to_eat.alive = False
            self.model.grid.remove_agent(sheep_to_eat)
            self.model.agents.remove(sheep_to_eat)

class Sheep(Animal):

    def __init__(self, model):
        super().__init__(model)
        self.alive = True
    def step(self):
        if self.alive:
            self.move()

