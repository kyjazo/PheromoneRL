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

# Funzione per generare una matrice HTML dei feromoni
#@solara.component
#def PheromoneMatrix(model, pheromone_type="wolf"):
#    data = model.get_pheromone_map(pheromone_type)
#    rows, cols = data.shape
#
#    # Crea una tabella HTML con i valori dei feromoni
#    return solara.HTML(
#        "table",
#        style={"border-collapse": "collapse"},
#        *[
#            solara.HTML(
#                "tr",
#                key=f"row-{i}",
#                *[
#                    solara.HTML(
#                        "td",
#                        key=f"cell-{i}-{j}",
#                        style={"border": "1px solid black", "padding": "5px", "text-align": "center"},
#                        children=[str(round(data[i, j], 2))],
#                    )
#                    for j in range(cols)
#                ],
#            )
#            for i in range(rows)
#        ],
#    )


model_params = {
    "width": 20,
    "height": 20,
    "initial_wolves": 5,
    "initial_sheep": 20,
}


myModel = WolfSheepModel(**model_params)


SpaceGraph = make_space_component(agent_portrayal)


#PheromoneMatrixWolf = lambda model: PheromoneMatrix(model, "wolf")
#PheromoneMatrixSheep = lambda model: PheromoneMatrix(model, "sheep")


viz = SolaraViz(
    model=myModel,
    components=[SpaceGraph],#, PheromoneMatrixWolf, PheromoneMatrixSheep],
    model_params=model_params,
    name="WolfSheepModel"
)


@solara.component
def Page():
    return viz