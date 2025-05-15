import solara
from mesa.visualization import SolaraViz, make_space_component, Slider
from model import WolfSheepModel
from agents import Wolf, Sheep, Pheromones


def agent_portrayal(agent):
    portrayal = {
        "size": 25,
    }

    if isinstance(agent, Wolf):
        portrayal["color"] = "darkred"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2

    elif isinstance(agent, Sheep):
        portrayal["color"] = "green"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2

    elif isinstance(agent, Pheromones):

       if agent.pheromone.wolf_concentration == 0 and agent.pheromone.sheep_concentration == 0:
           portrayal["color"] = "white"
           portrayal["marker"] = "s"
           portrayal["size"] = 75
       else:
           max_pheromone = max(agent.pheromone.wolf_concentration, agent.pheromone.sheep_concentration)
           red_intensity = (agent.pheromone.wolf_concentration / max_pheromone) if max_pheromone != 0 else 0
           green_intensity = (agent.pheromone.sheep_concentration / max_pheromone) if max_pheromone != 0 else 0


           red_hex = int(red_intensity * 255)
           green_hex = int(green_intensity * 255)
           portrayal["color"] = f"#{red_hex:02x}{green_hex:02x}00"
           portrayal["marker"] = "s"
           portrayal["size"] = 75

    return portrayal


model_params = {
    "render_pheromone": {
        "type": "Select",
        "value": False,
        "values": [True, False],
        "label": "Render Pheromone?",
    },
    "respawn": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "respawn?",
    },
    "learning": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "learning?",
    },

    "height": Slider("Height", 20, 5, 100, 5, dtype=int),
    "width": Slider("Width", 20, 5, 100, 5, dtype=int),
    "initial_sheep": Slider("Initial Sheep Population", 20, 1, 100, 1, dtype=int),
    "initial_wolves": Slider("Initial Wolf Population", 5, 1, 20, 1, dtype=int),
    "pheromone_evaporation": Slider("Pheromone Evaporation", 0.1, 0, 1, 0.01, dtype=float),
    "pheromone_added": Slider("Pheromone Released", 0.5, 0, 5, 0.1, dtype=float),
    "diffusion_rate": Slider("Diffusion Rate", 0.1, 0.01, 1, 0.1, dtype=float),

    "q_learning_params": {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.3,
        "epsilon_decay": 0.895,
        "min_epsilon": 0.00,
        "actions": [0, 1, 3] #eliminata azione 2
    },

}

myModel = WolfSheepModel()

# Aggiungi propertylayer_portrayal al make_space_component
SpaceGraph = make_space_component(
    agent_portrayal=agent_portrayal,
)

viz = SolaraViz(
    model=myModel,
    components=[SpaceGraph],
    model_params=model_params,
    name="WolfSheepModel"
)

@solara.component
def Page():
    return viz