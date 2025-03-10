import solara
import numpy as np
import plotly.graph_objects as go
from mesa.visualization import SolaraViz, make_space_component, Slider
from model import WolfSheepModel
from agents import Wolf, Sheep

def agent_portrayal(agent):
    portrayal = {
        "size": 50,
    }
    if isinstance(agent, Wolf):
        portrayal["color"] = "red"
        portrayal["shape"] = "circle"
        portrayal["r"] = 0.5
    elif isinstance(agent, Sheep):
        portrayal["color"] = "green"
        portrayal["shape"] = "circle"
        portrayal["r"] = 0.5

    return portrayal


model_params = {
    "height": Slider("Height", 20, 5, 100, 5, dtype=int),
    "width": Slider("Width", 20, 5, 100, 5, dtype=int),
    "initial_sheep": Slider("Initial Sheep Population", 5, 1, 10, 1, dtype=int),
    "initial_wolves": Slider("Initial Wolf Population", 20, 1, 100, 1, dtype=int),
    "pheromone_evaporation": Slider("Pheromone Evaporation", 0.1, 0, 1, 0.1, dtype=float),
    "pheromone_added": Slider("Pheromone Released", 0.5, 0, 5, 0.1, dtype=float),
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