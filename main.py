import mesa
import pandas as pd
import matplotlib.pyplot as plt
from model import WolfSheepModel
import os
import numpy as np
import json
from datetime import datetime
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from agents import QLearning
import multiprocessing





def save_q_table_to_results(q_table_file, output_dir):

    if os.path.exists(q_table_file):
        dest_path = os.path.join(output_dir, os.path.basename(q_table_file))
        shutil.copy2(q_table_file, dest_path)
        print(f"📋 Q-table salvata in: {dest_path}")
    else:
        print("⚠️ File Q-table non trovato, nessuna copia effettuata")
def get_next_test_folder(testing=False, learning=False, prefix="test"):
    global save  # Se stai usando `save` come variabile globale

    # Determina la cartella base corretta
    if testing and learning:
        base_dir = "test_results"
    elif testing and not learning:
        base_dir = "test_results_NL"
    else:
        base_dir = "results"

    os.makedirs(base_dir, exist_ok=True)

    # Ottieni tutte le directory che iniziano con il prefisso
    existing = [
        d for d in os.listdir(base_dir)
        if isinstance(d, str) and os.path.isdir(os.path.join(base_dir, d)) and d.startswith(str(prefix))
    ]

    # Estrai il numero dopo il prefisso, se presente
    numbers = [int(d[len(prefix):]) for d in existing if d[len(prefix):].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1

    # Costruisci il nuovo percorso
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


def plot_results(df, learning=True, output_dir="./results", window_size=100):
    plt.figure(figsize=(20, 8))
    plt.style.use('seaborn-v0_8')
    window_size = window_size

    if learning:
        reward_stats = df.groupby('iteration')['Reward'].agg(['mean', 'std']).reset_index()
        reward_stats['mean_rolling'] = reward_stats['mean'].rolling(window=window_size).mean()

        plt.subplot(1, 2, 1)
        plt.plot(reward_stats['iteration'], reward_stats['mean_rolling'],
                 color='darkblue', linewidth=2.5)

        if 'run_id' in df.columns and len(df['run_id'].unique()) > 1:
            reward_stats['std_rolling'] = reward_stats['std'].rolling(window=window_size).mean()
            plt.fill_between(reward_stats['iteration'],
                             reward_stats['mean_rolling'] - reward_stats['std_rolling'],
                             reward_stats['mean_rolling'] + reward_stats['std_rolling'],
                             color='blue', alpha=0.2)

        plt.title(f'Reward Media (rolling mean {window_size})', fontsize=16)
        plt.xlabel('Iterazione', fontsize=14)
        plt.ylabel('Reward', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.locator_params(axis='x', nbins=20)
        plt.subplot(1, 2, 2)

    if 'run_id' in df.columns:
        sheep_by_run = df.groupby(["run_id", "iteration"])["Sheep_eaten"].sum().reset_index()
        sheep_stats = sheep_by_run.groupby("iteration")["Sheep_eaten"].agg(["mean", "std"]).reset_index()
    else:
        sheep_stats = df.groupby("iteration")["Sheep_eaten"].agg(["mean"]).reset_index()
        sheep_stats["std"] = 0

    sheep_stats['mean_rolling'] = sheep_stats['mean'].rolling(window=window_size).mean()
    sheep_stats['std_rolling'] = sheep_stats['std'].rolling(window=window_size).mean()

    plt.plot(sheep_stats['iteration'], sheep_stats['mean_rolling'],
             color='red', linewidth=2.5)
    plt.fill_between(sheep_stats['iteration'],
                     sheep_stats['mean_rolling'] - sheep_stats['std_rolling'],
                     sheep_stats['mean_rolling'] + sheep_stats['std_rolling'],
                     color='red', alpha=0.2)
    plt.title(f'Pecore Mangiate (rolling mean {window_size})', fontsize=16)
    plt.xlabel('Iterazione', fontsize=14)
    plt.ylabel('Numero di Pecore', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        filename = f"reward_and_sheep_rolling{window_size}.png" if learning else f"sheep_only_rolling{window_size}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_simulation_steps(df, output_dir="./results", window_size=100):
    if save:
        os.makedirs(output_dir, exist_ok=True)
    if "Steps" not in df.columns:
        print("La colonna 'Steps' non è disponibile nei dati.")
        return

    window_size = window_size

    if 'run_id' in df.columns:
        steps_data = df.groupby(["run_id", "iteration"])["Steps"].max().reset_index()
        steps_stats = steps_data.groupby("iteration")["Steps"].agg(["mean", "std"]).reset_index()
    else:
        steps_stats = df.groupby("iteration")["Steps"].max().reset_index()
        steps_stats["std"] = 0

    steps_stats['mean_rolling'] = steps_stats['mean'].rolling(window=window_size).mean()
    steps_stats['std_rolling'] = steps_stats['std'].rolling(window=window_size).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(steps_stats["iteration"], steps_stats["mean_rolling"],
             color="green", linewidth=2.5,
             label=f"Numero di Step (rolling mean {window_size})")
    plt.fill_between(steps_stats["iteration"],
                     steps_stats["mean_rolling"] - steps_stats["std_rolling"],
                     steps_stats["mean_rolling"] + steps_stats["std_rolling"],
                     color='green', alpha=0.2)
    plt.title("Durata delle Simulazioni (in Step)", fontsize=16)
    plt.xlabel("Numero Simulazione", fontsize=14)
    plt.ylabel("Numero di Step", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout(pad=3)

    filepath = os.path.join(output_dir, f"steps_plot_rolling{window_size}.png")
    if save:
        plt.savefig(filepath)
        print(f"📈 Grafico steps salvato in: {filepath}")
    plt.show()


def plot_all_actions_in_one(df, output_dir="./results", window_size=100):
    plt.figure(figsize=(20, 10))
    plt.style.use('seaborn-v0_8')

    actions = ["Action_0", "Action_1", "Action_2", "Action_3"]
    action_labels = ["Follow sheep pheromone", "Random movement", "Release pheromone", "Walk away wolf pheromone"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    window_size = window_size

    if 'run_id' in df.columns:
        grouped = df.groupby(["run_id", "iteration"])[actions].sum().reset_index()
        mean_grouped = grouped.groupby("iteration")[actions].mean().reset_index()
        std_grouped = grouped.groupby("iteration")[actions].std().reset_index()
    else:
        mean_grouped = df.groupby("iteration")[actions].sum().reset_index()
        std_grouped = mean_grouped.copy()
        for action in actions:
            std_grouped[action] = 0

    for action, label, color in zip(actions, action_labels, colors):
        mean_grouped[f'{action}_rolling'] = mean_grouped[action].rolling(window=window_size).mean()
        std_grouped[f'{action}_rolling'] = std_grouped[action].rolling(window=window_size).mean()

        plt.plot(mean_grouped["iteration"], mean_grouped[f'{action}_rolling'],
                 label=label, linewidth=3, color=color, alpha=0.8)

        plt.fill_between(mean_grouped["iteration"],
                         mean_grouped[f'{action}_rolling'] - std_grouped[f'{action}_rolling'],
                         mean_grouped[f'{action}_rolling'] + std_grouped[f'{action}_rolling'],
                         color=color, alpha=0.2)

    plt.title(f"Distribuzione delle Azioni (rolling mean {window_size})", fontsize=18, pad=20)
    plt.xlabel("Iterazione", fontsize=16, labelpad=15)
    plt.ylabel("Conteggio Azioni", fontsize=16, labelpad=15)
    plt.legend(title="Azioni", title_fontsize=14, fontsize=12,
               frameon=True, shadow=True, facecolor='white', edgecolor='gray',
               bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, f"all_actions_usage_rolling{window_size}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()


def plot_capture_median(df, output_dir="./results", window_size=100):
    if "Capture_Intervals" not in df.columns:
        print("⚠️ Colonna 'Capture_Intervals' mancante.")
        return

    window_size = window_size

    df_valid = df[df["Capture_Intervals"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    exploded = df_valid.explode("Capture_Intervals")
    exploded["Capture_Intervals"] = pd.to_numeric(exploded["Capture_Intervals"], errors="coerce")

    if 'run_id' in df.columns:
        median_data = exploded.groupby(["run_id", "iteration"])["Capture_Intervals"].median().reset_index()
        stats = median_data.groupby("iteration")["Capture_Intervals"].agg(['mean', 'std']).reset_index()
    else:
        stats = exploded.groupby("iteration")["Capture_Intervals"].agg(['median']).reset_index()
        stats.columns = ["iteration", "mean"]
        stats["std"] = 0

    stats['mean_rolling'] = stats['mean'].rolling(window=window_size).mean()
    stats['std_rolling'] = stats['std'].rolling(window=window_size).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(stats["iteration"], stats["mean_rolling"],
             color='purple', linewidth=2.5)
    plt.fill_between(stats["iteration"],
                     stats["mean_rolling"] - stats["std_rolling"],
                     stats["mean_rolling"] + stats["std_rolling"],
                     color='purple', alpha=0.2)
    plt.title(f"Mediana Step tra Pecore Mangiate (rolling mean {window_size})", fontsize=16)
    plt.xlabel("Iterazione", fontsize=14)
    plt.ylabel("Step tra catture", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, f"capture_median_rolling{window_size}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"📈 Grafico mediana catture salvato in: {filepath}")
    plt.show()


def run_single_simulation(run_id, base_params, q_learning_params):
    try:

        params = base_params.copy()
        if not base_params['testing']:
            q_table_file = f"q_table_run_{run_id}.json"

            params["q_table_file"] = q_table_file
            q = QLearning(**q_learning_params, q_table_file=q_table_file)
        else:
            q_table_file = f"q_table_run_{run_id}.json"
            q = QLearning(**q_learning_params, q_table_file=params['q_table_file'])


        result = mesa.batch_run(
            lambda **kwargs: WolfSheepModel(**kwargs, q_learning=q),
            parameters={k: v for k, v in params.items() if k != "q_learning_params"},
            data_collection_period=-1,
            iterations=5000,
            number_processes=1,
            display_progress=True
        )

        return result, q_table_file

    except Exception as e:
        import traceback
        print(f"Errore nella simulazione {run_id}:\n", traceback.format_exc())
        raise

def merge_q_tables(q_table_files, output_file="q_table_avg.json"):
    combined_q_table = {}
    print(q_table_files)
    for file in q_table_files:
        with open(file, "r") as f:
            q_table = json.load(f)

        for state, actions in q_table.items():
            state = eval(state)
            if state not in combined_q_table:
                combined_q_table[state] = {int(a): float(v) for a, v in actions.items()}
            else:
                for a, v in actions.items():
                    a = int(a)
                    if a in combined_q_table[state]:
                        combined_q_table[state][a] += float(v)
                    else:
                        combined_q_table[state][a] = float(v)


    for state in combined_q_table:
        for action in combined_q_table[state]:
            combined_q_table[state][action] /= len(q_table_files)


    with open(output_file, "w") as f:
        json.dump({str(k): v for k, v in combined_q_table.items()}, f, indent=2)
    print(f"✅ Q-table media salvata in {output_file}")

def clean_up_q_tables(q_table_files, keep_file="q_table_avg.json"):
    for file in q_table_files:
        try:
            if os.path.exists(file) and file != keep_file:
                os.remove(file)
                print(f"🗑️ Q-table eliminata: {file}")
        except Exception as e:
            print(f"⚠️ Errore nell'eliminazione di {file}: {e}")




if __name__ == "__main__":

    multiprocessing.set_start_method("spawn", force=True)

    num_parallel_runs = 3

    base_params = {
        "width": 45,
        "height": 45,
        "initial_wolves": 10,
        "initial_sheep": 20,
        "learning": True,
        "max_steps": 200,
        "respawn": False,
        "diffusion_rate": 0.5,
        "pheromone_evaporation": 0.1,
        "testing": False,
        "q_table_file": "./ServerTest/test14/q_table_avg.json"
    }

    q_learning_params = {
        "actions": [0, 1, 3],
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 0.9985,
        "min_epsilon": 0.01
    }



    all_results = []
    q_tables_paths = []
    save = True


    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_simulation, i, base_params, q_learning_params)
                   for i in range(num_parallel_runs)]

        for i, future in enumerate(as_completed(futures)):
            sim_result, q_table_file = future.result()
            for r in sim_result:
                r["run_id"] = i
            all_results.extend(sim_result)
            if base_params['learning'] and not base_params['testing']:
                q_tables_paths.append(q_table_file)

    if base_params['learning'] and not base_params['testing']:
        merge_q_tables(q_tables_paths, output_file="q_table_avg.json")
        clean_up_q_tables(q_tables_paths, keep_file="q_table_avg.json")
    df = pd.DataFrame(all_results)


    print(df.dropna(subset=['Capture_Intervals']))
    df = df.dropna(subset=['Sheep_eaten'])

    #print(df['Reward'])
    #df['Reward'].to_csv("dataframe.csv")

    output_dir, abs_output_dir = get_next_test_folder(base_params['testing'], base_params['learning'])
    if save:
        save_simulation_metadata(base_params, q_learning_params, output_dir=output_dir)
        save_q_table_to_results("q_table_avg.json", abs_output_dir)
    plot_results(df, learning=True, output_dir=output_dir, window_size=100)
    plot_simulation_steps(df, output_dir=output_dir, window_size=100)
    plot_all_actions_in_one(df, output_dir=output_dir, window_size=100)
    plot_capture_median(df, output_dir=output_dir, window_size=100)








