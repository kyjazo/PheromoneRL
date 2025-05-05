import mesa
import pandas as pd
import matplotlib.pyplot as plt
from model import WolfSheepModel
import os
import numpy as np
import json
from datetime import datetime
import shutil

save = True

def save_q_table_to_results(q_table_file, output_dir):

    if os.path.exists(q_table_file):
        dest_path = os.path.join(output_dir, os.path.basename(q_table_file))
        shutil.copy2(q_table_file, dest_path)
        print(f"📋 Q-table salvata in: {dest_path}")
    else:
        print("⚠️ File Q-table non trovato, nessuna copia effettuata")

def get_next_test_folder(base_dir="results", prefix="test"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]

    numbers = [int(d[len(prefix):]) for d in existing if d[len(prefix):].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1


    new_folder = os.path.join(base_dir, f"{prefix}{next_number}")
    if save:
        os.makedirs(new_folder)
    return new_folder, os.path.abspath(new_folder)


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
    plt.figure(figsize=(20, 8))
    plt.style.use('seaborn-v0_8')

    if learning:
        reward_stats = df.groupby('iteration')['Reward'].agg(['mean', 'std']).reset_index()
        plt.subplot(1, 2, 1)
        plt.plot(reward_stats['iteration'], reward_stats['mean'], color='darkblue', linewidth=2.5)
        plt.fill_between(reward_stats['iteration'],
                         reward_stats['mean'] - reward_stats['std'],
                         reward_stats['mean'] + reward_stats['std'],
                         color='blue', alpha=0.2)
        plt.title('Reward Media', fontsize=16)
        plt.xlabel('Iterazione', fontsize=14)
        plt.ylabel('Reward', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.locator_params(axis='x', nbins=20)  # Riduci i tick

        plt.subplot(1, 2, 2)

    sheep_eaten = df.groupby('iteration')['Sheep_eaten'].sum().reset_index()
    plt.plot(sheep_eaten['iteration'], sheep_eaten['Sheep_eaten'], color='red', linewidth=2.5)
    plt.title('Pecore Mangiate', fontsize=16)
    plt.xlabel('Iterazione', fontsize=14)
    plt.ylabel('Numero di Pecore', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)  # Ruota i tick se sono troppi
    plt.tight_layout()

    filename = "reward_and_sheep.png" if learning else "sheep_only.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
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
    if save:
        plt.savefig(filepath)
        print(f"📈 Grafico steps salvato in: {filepath}")
    plt.show()

    print("\nAzioni totali per iterazione:")
    print(df.groupby("iteration")[["Action_0", "Action_1", "Action_2", "Action_3"]].sum().to_string())

#def plot_all_actions_in_one(df, output_dir="./results"):
#    os.makedirs(output_dir, exist_ok=True)
#
#    actions = ["Follow sheep pheromone", "Random movement", "Stay still", "Walk away wolf pheromone"]
#    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blu, arancione, verde, rosso
#
#    grouped = df.groupby("iteration")[actions].sum().reset_index()
#
#    plt.figure(figsize=(10, 6))
#
#    for action, color in zip(actions, colors):
#        plt.plot(grouped["iteration"], grouped[action],
#                 label=f"Azione {action[-1]}",
#                 linewidth=2.5, marker='o', markersize=5, color=color)
#
#    plt.title("Evoluzione dell'Uso delle Azioni", fontsize=15, pad=10)
#    plt.xlabel("Iterazione", fontsize=12)
#    plt.ylabel("Conteggio", fontsize=12)
#    plt.grid(True, linestyle=':', alpha=0.7)
#    plt.legend(fontsize=10)
#    plt.tight_layout()
#
#    filepath = os.path.join(output_dir, "all_actions_usage.png")
#    if save:
#        plt.savefig(filepath)
#        print(f"📊 Grafico unico azioni salvato in: {filepath}")
#    plt.show()

def plot_all_actions_in_one(df, output_dir="./results"):
    plt.figure(figsize=(20, 10))  # Grafico più grande (24 pollici orizzontali)
    plt.style.use('seaborn-v0_8')  # Stile moderno

    # Definizione azioni e relative etichette
    actions = ["Action_0", "Action_1", "Action_2", "Action_3"]
    action_labels = ["Follow sheep pheromone", "Random movement", "Stay still", "Walk away wolf pheromone"]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Colori distintivi (blu, arancio, verde, rosso)

    # Preparazione dati
    grouped = df.groupby("iteration")[actions].sum().reset_index()

    # Plot di ogni azione con stile migliorato
    for action, label, color in zip(actions,action_labels, colors):
        plt.plot(
            grouped["iteration"],
            grouped[action],
            label=label,
            linewidth=3,  # Linea più spessa
            color=color,
            marker='o',  # Aggiunta di marker
            markersize=8,
            markeredgecolor='black',
            alpha=0.8  # Leggera trasparenza
        )

    # Titoli e assi
    plt.title("Distribuzione delle Azioni per Iterazione", fontsize=18, pad=20)
    plt.xlabel("Iterazione", fontsize=16, labelpad=15)
    plt.ylabel("Conteggio Azioni", fontsize=16, labelpad=15)

    # Legenda avanzata
    legend = plt.legend(
        title="Azioni",
        title_fontsize=14,
        fontsize=12,
        frameon=True,
        shadow=True,
        facecolor='white',
        edgecolor='gray',
        bbox_to_anchor=(1.02, 1),  # Sposta la legenda a destra
        loc='upper left'
    )

    # Griglia e tick
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12, rotation=45)  # Tick ruotati per leggibilità
    plt.yticks(fontsize=12)

    # Ottimizzazione layout
    plt.tight_layout()

    # Salvataggio ad alta risoluzione
    filepath = os.path.join(output_dir, "all_actions_usage.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
def run_test_simulation(q_table_file, output_dir="./test_results", learning=True):

    if not learning:
        output_dir="./test_results_NL"

    os.makedirs(output_dir, exist_ok=True)


    params = {
        "width": 20,
        "height": 20,
        "initial_wolves": 5,
        "initial_sheep": 20,
        "q_table_file": q_table_file,
        "learning": learning,
        "testing": True,
        "max_steps": 200,
        "respawn": True,
    }


    result = mesa.batch_run(
        WolfSheepModel,
        parameters={k: v for k, v in params.items() if k != "q_learning_params"},
        data_collection_period=-1,
        iterations=200,
        display_progress=True,
        number_processes=1
    )

    # Salva i risultati
    df = pd.DataFrame(result)
    plot_results(df, learning=False, output_dir=output_dir)
    plot_simulation_steps(df, output_dir=output_dir)
    plot_all_actions_in_one(df, output_dir=output_dir)

    print("✅ Simulazione di test completata!")



if __name__ == "__main__":

    q_learning_params = {
        "actions": [0, 1, 2, 3],
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon": 0.5,
        "epsilon_decay": 0.965,
        "min_epsilon": 0.01
    }

    testing = True

    params = {"width": 20, "height": 20, "initial_wolves": 5, "initial_sheep": 20, "q_table_file": "q_table.json",
              "learning": True, "max_steps": 200, "respawn": True, "diffusion_rate": 0.5, "pheromone_evaporation": 0.1,
              "q_learning_params": q_learning_params}


    if testing:
        run_test_simulation("./results/test25/q_table.json", learning=False)

    else:
        result = mesa.batch_run(
            lambda **kwargs: WolfSheepModel(**kwargs, q_learning_params=q_learning_params),
            parameters={k: v for k, v in params.items() if k != "q_learning_params"},
            data_collection_period=-1,
            iterations=1000,
            display_progress=True,
            number_processes=1
        )


        df = pd.DataFrame(result)



        df = df.dropna(subset=['Sheep_eaten'])

        print(df.head)

        if "Steps" in df.columns:
            print("\nMedia sheep eaten per iterazione:")
            print(df.groupby("iteration")["Sheep_eaten"].mean())
        else:
            print("Colonna 'Steps' non trovata nei dati della simulazione.")



        output_dir, abs_output_dir = get_next_test_folder()

        #analyze_simulation(df)

        print("\nAzioni totali per iterazione:")
        print(df.groupby("iteration")[["Action_0", "Action_1", "Action_2", "Action_3"]].sum().to_string())

        # Salva i risultati
        if save:
            save_simulation_metadata(params, q_learning_params, output_dir=output_dir)


            # Copia il file Q-table nella cartella dei risultati
            save_q_table_to_results(params["q_table_file"], abs_output_dir)
        plot_results(df, learning=params["learning"], output_dir=output_dir)
        plot_simulation_steps(df, output_dir=output_dir)
        plot_all_actions_in_one(df, output_dir=output_dir)










