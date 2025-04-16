import mesa
import pandas as pd
import matplotlib.pyplot as plt
from model import WolfSheepModel
import os
import numpy as np
import json
from datetime import datetime


def get_next_test_folder(base_dir="results", prefix="test"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]

    numbers = [int(d[len(prefix):]) for d in existing if d[len(prefix):].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1

    new_folder = os.path.join(base_dir, f"{prefix}{next_number}")
    os.makedirs(new_folder)
    return new_folder


def save_simulation_metadata(params, learning_params=None, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "simulation_params": params
    }

    if learning_params:
        metadata["q_learning_params"] = learning_params

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"simulation_params_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Parametri salvati in {filepath}")


def analyze_simulation(df):
    print("\n=== Performance Metrics ===")
    # Mostra le pecore mangiate per ogni iterazione
    sheep_per_iteration = df.groupby('iteration')['Sheep_eaten'].sum()
    print("\nSheep eaten per iteration:")
    print(sheep_per_iteration.to_string())

def plot_results(df, learning=True, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    sheep_eaten = df.groupby('iteration')['Sheep_eaten'].sum().reset_index()

    if learning:
        reward_stats = df.groupby('iteration')['Reward'].agg(['mean', 'std']).reset_index()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(reward_stats['iteration'], reward_stats['mean'], color='#1f77b4', linewidth=2.5,
                 marker='o', markersize=5, markeredgecolor='darkblue')
        plt.fill_between(reward_stats['iteration'],
                         reward_stats['mean'] - reward_stats['std'],
                         reward_stats['mean'] + reward_stats['std'],
                         color='#1f77b4', alpha=0.15)
        plt.title('Evoluzione della Reward Media')
        plt.xlabel('Numero Simulazione')
        plt.ylabel('Valore Reward')
        plt.grid(True)
        plt.subplot(1, 2, 2)
    else:
        plt.figure(figsize=(7, 6))

    plt.plot(sheep_eaten['iteration'], sheep_eaten['Sheep_eaten'],
             color='#ff7f0e', linewidth=2.5,
             marker='s', markersize=5, markeredgecolor='#d62728')
    plt.title('Andamento Pecore Mangiate')
    plt.xlabel('Numero Simulazione')
    plt.ylabel('Numero di Pecore')
    plt.grid(True)
    plt.tight_layout(pad=3)

    filename = "reward_and_sheep.png" if learning else "sheep_only.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"📊 Grafico salvato in: {filepath}")
    plt.show()


def plot_simulation_steps(df, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    if "Steps" not in df.columns:
        print("La colonna 'Steps' non è disponibile nei dati.")
        return

    steps_per_simulation = df.groupby("iteration")["Steps"].max().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(steps_per_simulation["iteration"], steps_per_simulation["Steps"],
             color="green", linewidth=2.5, marker='^', markersize=6,
             label="Numero di Step per Simulazione")
    plt.title("Durata delle Simulazioni (in Step)")
    plt.xlabel("Numero Simulazione")
    plt.ylabel("Numero di Step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout(pad=3)

    filepath = os.path.join(output_dir, "steps_plot.png")
    plt.savefig(filepath)
    print(f"📈 Grafico steps salvato in: {filepath}")
    plt.show()

    print("\nAzioni totali per iterazione:")
    print(df.groupby("iteration")[["Action_0", "Action_1", "Action_2", "Action_3"]].sum().to_string())

def plot_all_actions_in_one(df, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)

    actions = ["Action_0", "Action_1", "Action_2", "Action_3"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blu, arancione, verde, rosso

    grouped = df.groupby("iteration")[actions].sum().reset_index()

    plt.figure(figsize=(10, 6))

    for action, color in zip(actions, colors):
        plt.plot(grouped["iteration"], grouped[action],
                 label=f"Azione {action[-1]}",
                 linewidth=2.5, marker='o', markersize=5, color=color)

    plt.title("Evoluzione dell'Uso delle Azioni", fontsize=15, pad=10)
    plt.xlabel("Iterazione", fontsize=12)
    plt.ylabel("Conteggio", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "all_actions_usage.png")
    plt.savefig(filepath)
    print(f"📊 Grafico unico azioni salvato in: {filepath}")
    plt.show()




if __name__ == "__main__":


    q_learning_params = {
        "actions": [0, 1, 2, 3],
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.3,
        "epsilon_decay": 0.895,
        "min_epsilon": 0.00
    }

    params = {"width": 20, "height": 20, "initial_wolves": 5, "initial_sheep": 20, "q_table_file": "q_table.json",
              "learning": False, "max_steps": 100, "respawn": False, "q_learning_params": None}

    result = mesa.batch_run(
        WolfSheepModel,
        data_collection_period=100,
        parameters=params,
        iterations=100,
        display_progress=True,
        number_processes=1
    )
    # Estrai i modelli dai risultati
    models = [r for r in result if isinstance(r, WolfSheepModel)]

    df = pd.DataFrame(result)



    df = df.dropna(subset=['Sheep_eaten'])

    if "Steps" in df.columns:
        print("\nMedia sheep eaten per iterazione:")
        print(df.groupby("iteration")["Sheep_eaten"].mean())
    else:
        print("Colonna 'Steps' non trovata nei dati della simulazione.")

    output_dir = get_next_test_folder()  # es: results/test3

    #analyze_simulation(df)

    print("\nAzioni totali per iterazione:")
    print(df.groupby("iteration")[["Action_0", "Action_1", "Action_2", "Action_3"]].sum().to_string())



    save_simulation_metadata(params, q_learning_params, output_dir=output_dir)
    plot_results(df, learning=params["learning"], output_dir=output_dir)
    plot_simulation_steps(df, output_dir=output_dir)
    plot_all_actions_in_one(df, output_dir=output_dir)









