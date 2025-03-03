import solara
from mesa.visualization import SolaraViz, make_space_component
from model import WolfSheepModel
from agents import Wolf, Sheep
import numpy as np

def agent_portrayal(agent):
    portrayal = {}
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
    "width": 20,
    "height": 20,
    "initial_wolves": 5,
    "initial_sheep": 20,
}

myModel = WolfSheepModel(**model_params)


SpaceGraph = make_space_component(agent_portrayal)

viz = SolaraViz(
    model=myModel,
    components=[SpaceGraph],
    #model_params=model_params,
    name="WolfSheepModel"
)

@solara.component
def Page():
    return viz