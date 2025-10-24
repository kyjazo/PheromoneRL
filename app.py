import solara
from mesa.visualization import SolaraViz, make_space_component, Slider
from model import WolfSheepModel
from agents import Wolf, Sheep, Pheromones, QLearning, Trail


def agent_portrayal(agent):
    portrayal = {
        "size": 100,
    }

    if isinstance(agent, Wolf):
        portrayal["color"] = "darkred"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2

    elif isinstance(agent, Sheep):
        portrayal["color"] = "green"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2

    #elif isinstance(agent, Pheromones):
##
    #   if agent.pheromone.wolf_concentration == 0 and agent.pheromone.sheep_concentration == 0:
    #       portrayal["color"] = "white"
    #       portrayal["marker"] = "s"
    #       portrayal["size"] = 75
    #   else:
    #       max_pheromone = max(agent.pheromone.wolf_concentration, agent.pheromone.sheep_concentration)
    #       red_intensity = (agent.pheromone.wolf_concentration / max_pheromone) if max_pheromone != 0 else 0
    #       green_intensity = (agent.pheromone.sheep_concentration / max_pheromone) if max_pheromone != 0 else 0
##
    #       alpha = 0
#
##
    #       red_hex = int(red_intensity * 255)
    #       green_hex = int(green_intensity * 255)
    #       portrayal["color"] = f"#{red_hex:02x}{green_hex:02x}00"
    #       portrayal["marker"] = "s"
    #       portrayal["size"] = 75
    elif isinstance(agent, Pheromones):
        if agent.pheromone.wolf_concentration == 0 and agent.pheromone.sheep_concentration == 0:
            portrayal["color"] = "white"
            portrayal["marker"] = "s"
            portrayal["size"] = 75
        else:
            max_pheromone = max(agent.pheromone.wolf_concentration, agent.pheromone.sheep_concentration)
            red_intensity = (agent.pheromone.wolf_concentration / max_pheromone) if max_pheromone != 0 else 0
            green_intensity = (agent.pheromone.sheep_concentration / max_pheromone) if max_pheromone != 0 else 0


            blend_to_white = 0.5


            red_val = int(red_intensity * 255)
            green_val = int(green_intensity * 255)
            blue_val = 0


            red_val = int(red_val + blend_to_white * (255 - red_val))
            green_val = int(green_val + blend_to_white * (255 - green_val))
            blue_val = int(blue_val + blend_to_white * (255 - blue_val))

            portrayal["color"] = f"#{red_val:02x}{green_val:02x}{blue_val:02x}"
            portrayal["marker"] = "s"
            portrayal["size"] = 75




    elif isinstance(agent, Trail):
        portrayal.update({
            "color": agent.get_rgb_color(),
            "marker": "o",
            "size": 25,
            "zorder": 1,
        })

    return portrayal

#q_learning_params = {
#        "actions": [0, 1, 2, 3, 4, 5],
#        "alpha": 0.01,
#        "gamma": 0.99,
#        "epsilon": 0.5,
#        "epsilon_decay": 0.9985,
#        "min_epsilon": 0.01
#    }
q_learning_params = {
    "actions": [0, 1, 2, 3, 4, 5],
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 0.5,
    "epsilon_decay": 0.9985,
    "min_epsilon": 0.01
}


q = QLearning(**q_learning_params)

model_params = {
    "render_pheromone": {
        "type": "Select",
        "value": False,
        "values": [True, False],
        "label": "Render Pheromone?",
    },
    "respawn": {
        "type": "Select",
        "value": False,
        "values": [True, False],
        "label": "respawn?",
    },
    "learning": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "learning?",
    },
    "testing": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "testing?",
    },
    "torus": {
        "type": "Select",
        "value": False,
        "values": [True, False],
        "label": "Torus?",
    },

    "height": Slider("Height", 45, 5, 100, 5, dtype=int),
    "width": Slider("Width", 45, 5, 100, 5, dtype=int),
    "initial_sheep": Slider("Initial Sheep Population", 20, 1, 100, 1, dtype=int),
    "initial_wolves": Slider("Initial Wolf Population", 10, 1, 20, 1, dtype=int),
    "pheromone_evaporation": Slider("Pheromone Evaporation", 0.1, 0, 1, 0.01, dtype=float),
    "pheromone_added": Slider("Pheromone Released", 1, 0, 20, 1, dtype=float),
    "diffusion_rate": Slider("Diffusion Rate", 0.1, 0.01, 1, 0.1, dtype=float),

    "q_learning": q


}

SpaceGraph = make_space_component(
    agent_portrayal=agent_portrayal,
    backend="matplotlib"

)
wolfmodel=WolfSheepModel()
viz = SolaraViz(
    model=wolfmodel,
    components=[SpaceGraph],
    model_params=model_params,
    name="WolfSheepModel"
)


@solara.component
def Page():
    return viz
