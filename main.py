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


    sheep_by_run = df.groupby(["run_id", "iteration"])["Sheep_eaten"].sum().reset_index()


    sheep_eaten = sheep_by_run.groupby("iteration")["Sheep_eaten"].mean().reset_index()


    plt.plot(sheep_eaten['iteration'], sheep_eaten['Sheep_eaten'], color='red', linewidth=2.5)
    plt.title('Pecore Mangiate', fontsize=16)
    plt.xlabel('Iterazione', fontsize=14)
    plt.ylabel('Numero di Pecore', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        filename = "reward_and_sheep.png" if learning else "sheep_only.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_simulation_steps(df, output_dir="./results"):

    if save:
        os.makedirs(output_dir, exist_ok=True)
    if "Steps" not in df.columns:
        print("La colonna 'Steps' non è disponibile nei dati.")
        return

    # Calcola il massimo per ogni iteration-run
    max_steps_each_run = df.groupby(["run_id", "iteration"])["Steps"].max().reset_index()

    # Ora fai la media tra le 5 run per ogni iteration
    steps_per_simulation = max_steps_each_run.groupby("iteration")["Steps"].mean().reset_index()

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
    print(df.groupby("iteration")[["Action_0", "Action_1", "Action_3"]].sum().to_string()) #eliminata azione 2

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
    plt.figure(figsize=(20, 10))
    plt.style.use('seaborn-v0_8')


    actions = ["Action_0", "Action_1", "Action_3"] #eliminata azione 2
    action_labels = ["Follow sheep pheromone", "Random movement", "Walk away wolf pheromone"]

    colors = ["#1f77b4", "#ff7f0e", "#d62728"]


    actions_per_run = df.groupby(["run_id", "iteration"])[
        ["Action_0", "Action_1", "Action_3"]].sum().reset_index()


    grouped = actions_per_run.groupby("iteration")[["Action_0", "Action_1", "Action_3"]].mean().reset_index()


    for action, label, color in zip(actions,action_labels, colors):
        plt.plot(
            grouped["iteration"],
            grouped[action],
            label=label,
            linewidth=3,
            color=color,
            marker='o',
            markersize=8,
            markeredgecolor='black',
            alpha=0.8
        )


    plt.title("Distribuzione delle Azioni per Iterazione", fontsize=18, pad=20)
    plt.xlabel("Iterazione", fontsize=16, labelpad=15)
    plt.ylabel("Conteggio Azioni", fontsize=16, labelpad=15)


    legend = plt.legend(
        title="Azioni",
        title_fontsize=14,
        fontsize=12,
        frameon=True,
        shadow=True,
        facecolor='white',
        edgecolor='gray',
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )


    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)


    plt.tight_layout()


    if save:
        filepath = os.path.join(output_dir, "all_actions_usage.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()


def run_test_simulation(q_table_file="q_table_avg.json", output_dir="./test_results", learning=True):

    if not learning:
        output_dir="./test_results_NL"

    os.makedirs(output_dir, exist_ok=True)


    params = {
        "width": 45,
        "height": 45,
        "initial_wolves": 5,
        "initial_sheep": 20,
        "q_table_file": q_table_file,
        "learning": learning,
        "testing": True,
        "max_steps": 100,
        "respawn": True,
    }


    result = mesa.batch_run(
        WolfSheepModel,
        parameters={k: v for k, v in params.items() if k != "q_learning_params"},
        data_collection_period=-1,
        iterations=100,
        display_progress=True,
        number_processes=1
    )


    df = pd.DataFrame(result)
    plot_results(df, learning=False, output_dir=output_dir)
    plot_simulation_steps(df, output_dir=output_dir)
    plot_all_actions_in_one(df, output_dir=output_dir)

    print("✅ Simulazione di test completata!")


#if __name__ == "__main__":
#
#    q_learning_params = {
#        "actions": [0, 1, 3], #eliminata azione 2
#        "alpha": 0.2,
#        "gamma": 0.9,
#        "epsilon": 0.5,
#        "epsilon_decay": 0.975,
#        "min_epsilon": 0.01
#    }
#
#    testing = False
#    save = False
#
#    params = {"width": 35, "height": 35, "initial_wolves": 5, "initial_sheep": 20, "q_table_file": "q_table.json",
#              "learning": True, "max_steps": 200, "respawn": False, "diffusion_rate": 0.5, "pheromone_evaporation": 0.1,
#              "q_learning_params": q_learning_params}
#
#
#    if testing:
#        run_test_simulation("./results/test36/q_table.json", learning=True)
#
#    else:
#        result = mesa.batch_run(
#            lambda **kwargs: WolfSheepModel(**kwargs, q_learning_params=q_learning_params),
#            parameters={k: v for k, v in params.items() if k != "q_learning_params"},
#            data_collection_period=-1,
#            iterations=100,
#            display_progress=True,
#            number_processes=1
#        )
#
#
#        df = pd.DataFrame(result)
#
#
#
#        df = df.dropna(subset=['Sheep_eaten'])
#
#        print(df.head)
#
#        if "Steps" in df.columns:
#            print("\nMedia sheep eaten per iterazione:")
#            print(df.groupby("iteration")["Sheep_eaten"].mean())
#        else:
#            print("Colonna 'Steps' non trovata nei dati della simulazione.")
#
#
#
#        output_dir, abs_output_dir = get_next_test_folder()
#
#        #analyze_simulation(df)
#
#        print("\nAzioni totali per iterazione:")
#        print(df.groupby("iteration")[["Action_0", "Action_1", "Action_3"]].sum().to_string())
#
#
#        if save:
#            save_simulation_metadata(params, q_learning_params, output_dir=output_dir)
#            save_q_table_to_results(params["q_table_file"], abs_output_dir)
#
#        plot_results(df, learning=params["learning"], output_dir=output_dir)
#        plot_simulation_steps(df, output_dir=output_dir)
#        plot_all_actions_in_one(df, output_dir=output_dir)


def run_single_simulation(run_id, base_params, q_learning_params):
    q_table_file = f"q_table_run_{run_id}.json"
    params = base_params.copy()
    params["q_table_file"] = q_table_file

    q = QLearning(**q_learning_params, q_table_file=q_table_file)
    print(q.epsilon)
    result = mesa.batch_run(
        lambda **kwargs: WolfSheepModel(**kwargs, q_learning=q),
        parameters={k: v for k, v in params.items() if k != "q_learning_params"},
        data_collection_period=-1,
        iterations=300,
        number_processes=1,
        display_progress=True
    )
    print(q.epsilon)
    return result, q_table_file

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

def plot_capture_median(df, output_dir="./results"):
    if "Capture_Intervals" not in df.columns:
        print("⚠️ Colonna 'Capture_Intervals' mancante.")
        return


    df_valid = df[df["Capture_Intervals"].apply(lambda x: isinstance(x, list) and len(x) > 0)]


    exploded = df_valid.explode("Capture_Intervals")


    exploded["Capture_Intervals"] = pd.to_numeric(exploded["Capture_Intervals"], errors="coerce")


    median_by_iter = exploded.groupby("iteration")["Capture_Intervals"].median().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(median_by_iter["iteration"], median_by_iter["Capture_Intervals"],
             marker='o', color='purple', linewidth=2.5)

    plt.title("Mediana Step tra Pecore Mangiate", fontsize=16)
    plt.xlabel("Iterazione", fontsize=14)
    plt.ylabel("Step tra catture", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, "capture_median.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"📈 Grafico mediana catture salvato in: {filepath}")
    plt.show()



if __name__ == "__main__":



    num_parallel_runs = 3

    base_params = {
        "width": 45,
        "height": 45,
        "initial_wolves": 5,
        "initial_sheep": 20,
        "learning": True,
        "max_steps": 1000,
        "respawn": True,
        "diffusion_rate": 0.5,
        "pheromone_evaporation": 0.1,
    }

    q_learning_params = {
        "actions": [0, 1, 3],
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 0.996,
        "min_epsilon": 0.01
    }



    all_results = []
    q_tables_paths = []
    testing = False
    save = True

    if testing:
        run_test_simulation(learning=False)

    else:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_single_simulation, i, base_params, q_learning_params)
                       for i in range(num_parallel_runs)]

            for i, future in enumerate(as_completed(futures)):
                sim_result, q_table_file = future.result()
                for r in sim_result:
                    r["run_id"] = i
                all_results.extend(sim_result)
                q_tables_paths.append(q_table_file)

        merge_q_tables(q_tables_paths, output_file="q_table_avg.json")
        clean_up_q_tables(q_tables_paths, keep_file="q_table_avg.json")

        df = pd.DataFrame(all_results)

        print(df)
        print(df.dropna(subset=['Capture_Intervals']))
        df = df.dropna(subset=['Sheep_eaten'])



        output_dir, abs_output_dir = get_next_test_folder()

        if save:
            save_simulation_metadata(base_params, q_learning_params, output_dir=output_dir)
            save_q_table_to_results("q_table_avg.json", abs_output_dir)



        plot_results(df, learning=True, output_dir=output_dir)
        plot_simulation_steps(df, output_dir=output_dir)
        plot_all_actions_in_one(df, output_dir=output_dir)
        plot_capture_median(df, output_dir=output_dir)












