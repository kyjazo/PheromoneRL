import gc

import mesa
import pandas as pd
import matplotlib.pyplot as plt
from model import WolfSheepModel
from agents import Wolf
import os
import json
from datetime import datetime
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from agents import QLearning
import multiprocessing
from collections import defaultdict
from memory_profiler import profile
import gc
import psutil
def check_memory():
    process = psutil.Process()
    mem = process.memory_info()
    print(f"[MEM] RSS: {mem.rss / 1024 / 1024:.2f} MB")
    print(f"[MEM] VMS: {mem.vms / 1024 / 1024:.2f} MB")
    #print(f"[MEM] Shared: {mem.shared / 1024 / 1024:.2f} MB")




#def save_q_table_to_results(q_table_file, output_dir):
#    """
#    Copia la cartella 'q_tables' (contenente q_table_0.json, q_table_1.json, ...) nella cartella di output.
#    Non viene salvata la q_table media.
#    """
#    if not q_table_file:
#        print("ℹ️ Nessun q_table_file specificato; nessuna q_table da copiare.")
#        return
#
#    q_tables_dir = os.path.join(os.path.dirname(os.path.abspath(q_table_file)), "q_tables")
#    if os.path.exists(q_tables_dir) and os.path.isdir(q_tables_dir):
#        dest_q_tables = os.path.join(output_dir, "q_tables")
#        if os.path.exists(dest_q_tables):
#            shutil.rmtree(dest_q_tables)
#        shutil.copytree(q_tables_dir, dest_q_tables)
#        print(f"📁 Cartella q_tables copiata in: {dest_q_tables}")
#    else:
#        print("ℹ️ Nessuna cartella 'q_tables' trovata da copiare.")
def save_q_table_to_results(run_dirs, output_dir, n_wolves):
    """
    Combina le q-tables di più run concorrenti e salva solo la media.
    """
    merge_q_tables(run_dirs, output_dir, n_wolves)


def get_next_test_folder(testing=False, learning=False, prefix="test"):
    global save


    if testing and learning:
        base_dir = "test_results"
    elif testing and not learning:
        base_dir = "test_results_NL"
    else:
        base_dir = "results"

    os.makedirs(base_dir, exist_ok=True)


    existing = [
        d for d in os.listdir(base_dir)
        if isinstance(d, str) and os.path.isdir(os.path.join(base_dir, d)) and d.startswith(str(prefix))
    ]


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

        plt.title(f'Average reward (rolling mean {window_size})', fontsize=16)
        plt.xlabel('Episode', fontsize=14)
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
    plt.title(f'Sheep Eaten (rolling mean {window_size})', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Number of Sheep', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        filename = f"reward_and_sheep_rolling{window_size}.png" if learning else f"sheep_only_rolling{window_size}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
def plot_reward(df, output_dir="./results", window_size=100):
    if "Reward" not in df.columns:
        print("⚠️ 'Reward' column not found.")
        return

    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8')

    reward_stats = df.groupby('iteration')['Reward'].agg(['mean', 'std']).reset_index()
    reward_stats['mean_rolling'] = reward_stats['mean'].rolling(window=window_size).mean()

    plt.plot(reward_stats['iteration'], reward_stats['mean_rolling'],
             color='darkblue', linewidth=2.5, label='Average Reward')

    if 'run_id' in df.columns and len(df['run_id'].unique()) > 1:
        reward_stats['std_rolling'] = reward_stats['std'].rolling(window=window_size).mean()
        plt.fill_between(reward_stats['iteration'],
                         reward_stats['mean_rolling'] - reward_stats['std_rolling'],
                         reward_stats['mean_rolling'] + reward_stats['std_rolling'],
                         color='blue', alpha=0.2)

    plt.title(f'Average Reward (rolling mean {window_size})', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save:
        filename = f"reward_rolling{window_size}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_sheep_eaten(df, output_dir="./results", window_size=100):
    if "Sheep_eaten" not in df.columns:
        print("⚠️ 'Sheep_eaten' column not found.")
        return

    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8')

    if 'run_id' in df.columns:
        grouped = df.groupby(["run_id", "iteration"])["Sheep_eaten"].sum().reset_index()
        stats = grouped.groupby("iteration")["Sheep_eaten"].agg(["mean", "std"]).reset_index()
    else:
        stats = df.groupby("iteration")["Sheep_eaten"].agg(["mean"]).reset_index()
        stats["std"] = 0

    stats['mean_rolling'] = stats['mean'].rolling(window=window_size).mean()
    stats['std_rolling'] = stats['std'].rolling(window=window_size).mean()

    plt.plot(stats['iteration'], stats['mean_rolling'],
             color='red', linewidth=2.5, label='Sheep Eaten')

    plt.fill_between(stats['iteration'],
                     stats['mean_rolling'] - stats['std_rolling'],
                     stats['mean_rolling'] + stats['std_rolling'],
                     color='red', alpha=0.2)

    plt.title(f'Sheep Eaten (rolling mean {window_size})', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Number of Sheep', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save:
        filename = f"sheep_eaten_rolling{window_size}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_simulation_steps(df, output_dir="./results", window_size=100):
    if save:
        os.makedirs(output_dir, exist_ok=True)
    if "Step" not in df.columns:
        print("La colonna 'Steps' non è disponibile nei dati.")
        return

    window_size = window_size

    if 'run_id' in df.columns:
        steps_data = df.groupby(["run_id", "iteration"])["Step"].max().reset_index()
        steps_stats = steps_data.groupby("iteration")["Step"].agg(["mean", "std"]).reset_index()
    else:
        steps_stats = df.groupby("iteration")["Step"].max().reset_index()
        steps_stats["std"] = 0

    steps_stats['mean_rolling'] = steps_stats['mean'].rolling(window=window_size).mean()
    steps_stats['std_rolling'] = steps_stats['std'].rolling(window=window_size).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(steps_stats["iteration"], steps_stats["mean_rolling"],
             color="green", linewidth=2.5,
             label=f"Number of Steps (rolling mean {window_size})")
    plt.fill_between(steps_stats["iteration"],
                     steps_stats["mean_rolling"] - steps_stats["std_rolling"],
                     steps_stats["mean_rolling"] + steps_stats["std_rolling"],
                     color='green', alpha=0.2)
    plt.title("Duration of Episodes in Steps", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Steps", fontsize=14)
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

    actions = ["Action_0", "Action_1", "Action_2", "Action_3","Action_4", "Action_5"]
    action_labels = ["Follow sheep pheromone", "Random movement", "Release pheromone", "Run away wolf pheromone",
                     "Follow + Deposit", "Run away + Deposit"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

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

    plt.title(f"Actions Distribution (rolling mean {window_size})", fontsize=18, pad=20)
    plt.xlabel("Episode", fontsize=16, labelpad=15)
    plt.ylabel("Action Count", fontsize=16, labelpad=15)
    plt.legend(title="Actions", title_fontsize=14, fontsize=12,
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


#def plot_capture_median(df, output_dir="./results", window_size=100):
#    if "Capture_Intervals" not in df.columns:
#        print("⚠️ Colonna 'Capture_Intervals' mancante.")
#        return
#
#    window_size = window_size
#
#    df_valid = df[df["Capture_Intervals"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
#    exploded = df_valid.explode("Capture_Intervals")
#    exploded["Capture_Intervals"] = pd.to_numeric(exploded["Capture_Intervals"], errors="coerce")
#
#    if 'run_id' in df.columns:
#        median_data = exploded.groupby(["run_id", "iteration"])["Capture_Intervals"].median().reset_index()
#        stats = median_data.groupby("iteration")["Capture_Intervals"].agg(['mean', 'std']).reset_index()
#    else:
#        stats = exploded.groupby("iteration")["Capture_Intervals"].agg(['median']).reset_index()
#        stats.columns = ["iteration", "mean"]
#        stats["std"] = 0
#
#    stats['mean_rolling'] = stats['mean'].rolling(window=window_size).mean()
#    stats['std_rolling'] = stats['std'].rolling(window=window_size).mean()
#
#    plt.figure(figsize=(10, 5))
#    plt.plot(stats["iteration"], stats["mean_rolling"],
#             color='purple', linewidth=2.5)
#    plt.fill_between(stats["iteration"],
#                     stats["mean_rolling"] - stats["std_rolling"],
#                     stats["mean_rolling"] + stats["std_rolling"],
#                     color='purple', alpha=0.2)
#    plt.title(f"Median Steps Between Captures (rolling mean {window_size})", fontsize=16)
#    plt.xlabel("Episode", fontsize=14)
#    plt.ylabel("Steps between captures", fontsize=14)
#    plt.grid(True, linestyle='--', alpha=0.6)
#    plt.tight_layout()
#
#    if save:
#        filepath = os.path.join(output_dir, f"capture_median_rolling{window_size}.png")
#        plt.savefig(filepath, dpi=300, bbox_inches='tight')
#        print(f"📈 Grafico mediana catture salvato in: {filepath}")
#    plt.show()
def plot_capture_median(df, output_dir="./results", window_size=100):
    if "Capture_Intervals" not in df.columns:
        print("⚠️ Colonna 'Capture_Intervals' mancante.")
        return

    if save:
        os.makedirs(output_dir, exist_ok=True)

    if 'run_id' in df.columns:
        median_data = df.groupby(["run_id", "iteration"])["Capture_Intervals"].median().reset_index()
        stats = median_data.groupby("iteration")["Capture_Intervals"].agg(['mean', 'std']).reset_index()
    else:
        stats = df.groupby("iteration")["Capture_Intervals"].agg(['mean']).reset_index()
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
    plt.title(f"Median Steps Between Captures (rolling mean {window_size})", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Median Steps", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, f"capture_median_rolling{window_size}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"📈 Grafico mediana catture salvato in: {filepath}")
    plt.show()

def run_single_simulation(run_id, base_params, q_learning_params, num_episodes=5000):

    try:
        params = base_params.copy()
        all_results = []

        if base_params["learning"] and not base_params["testing"]:
            run_output_dir = os.path.join("./tmp_runs", f"run_{run_id}")
            os.makedirs(run_output_dir, exist_ok=True)

            q_table_file = os.path.join(run_output_dir, "q_table_anchor.json")
            params["q_table_file"] = q_table_file
            q = QLearning(**q_learning_params, q_table_file=q_table_file)


        else:
            run_output_dir = None
            q_table_file = params.get("q_table_file", None)

            q = QLearning(**q_learning_params, q_table_file=q_table_file)


        for iteration in range(num_episodes):
            model = WolfSheepModel(
                **{k: v for k, v in params.items() if k != "q_learning_params"},
                q_learning=q
            )

            while model.running:
                model.step()

            agent_data = model.datacollector.get_agenttype_vars_dataframe(Wolf).reset_index()
            agent_data["iteration"] = iteration
            agent_data["run_id"] = run_id
            all_results.extend(agent_data.to_dict("records"))


            model.grid = None
            model.remove_all_agents()
            model.datacollector = None
            del model
            gc.collect()

            print("Episodio: ", iteration)


        if base_params["learning"] and not base_params["testing"]:
            q.save_q_table(q_table_file)
            return all_results, run_output_dir
        else:
            return all_results, None

    except Exception as e:
        import traceback
        print(f"⚠️ Errore in run {run_id}: {traceback.format_exc()}")
        return [], None


    except Exception as e:
        import traceback
        print(f"⚠️ Errore nella simulazione {run_id}: {traceback.format_exc()}")
        return [], None


def merge_q_tables(run_dirs, output_dir, n_wolves):

    q_tables_out_dir = os.path.join(output_dir, "q_tables")
    os.makedirs(q_tables_out_dir, exist_ok=True)

    for idx in range(n_wolves):
        combined_q = defaultdict(lambda: defaultdict(float))
        counts = defaultdict(lambda: defaultdict(int))

        for run_dir in run_dirs:
            q_file = os.path.join(run_dir, "q_tables", f"q_table_{idx}.json")
            if not os.path.exists(q_file):
                continue
            with open(q_file, "r") as f:
                q_table = json.load(f)

            for state, actions in q_table.items():
                for action, value in actions.items():
                    combined_q[state][action] += float(value)
                    counts[state][action] += 1

        # facciamo la media
        avg_q = {}
        for state, actions in combined_q.items():
            avg_q[state] = {}
            for action, total in actions.items():
                avg_q[state][action] = total / counts[state][action]

        out_file = os.path.join(q_tables_out_dir, f"q_table_{idx}.json")
        with open(out_file, "w") as f:
            json.dump(avg_q, f)
        print(f"[DEBUG] Q-table media per Wolf {idx} salvata in {out_file}")
def clean_up_q_tables(run_dirs):
    """
    Rimuove le cartelle temporanee di ogni run (es. ./tmp_runs/run_X).
    """
    for run_dir in run_dirs:
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
            print(f"[DEBUG] Rimossa cartella temporanea {run_dir}")



if __name__ == "__main__":

    multiprocessing.set_start_method("spawn", force=True)

    num_parallel_runs = 3
    #per primo esperimento num_wolves 5, torus true e controllare le azioni
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
        "q_table_file": None,
        "torus": False
    }

    q_learning_params = {
        "actions": [0, 1, 2, 3, 4, 5],
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_decay": 0.9985,
        "min_epsilon": 0.01
    }

    all_results = []
    run_dirs = []  # qui accumuliamo le cartelle temporanee delle run

    save = True


    try:
        with ProcessPoolExecutor(max_workers=num_parallel_runs) as executor:
            futures = [executor.submit(run_single_simulation, i, base_params, q_learning_params)
                       for i in range(num_parallel_runs)]

            for i, future in enumerate(as_completed(futures)):
                try:
                    sim_result, run_dir = future.result()

                    for r in sim_result:
                        r["run_id"] = i
                    all_results.extend(sim_result)

                    if base_params['learning'] and not base_params['testing'] and run_dir is not None:
                        run_dirs.append(run_dir)

                except Exception as e:
                    import traceback
                    print(f"⚠️ Errore nella simulazione {i}: {traceback.format_exc()}")

    except KeyboardInterrupt:
        print("⛔ Interrotto dall'utente.")

    print(run_dirs)

    if all_results:
        df = pd.DataFrame(all_results)
        df = df.dropna(subset=['Sheep_eaten'])

        output_dir, abs_output_dir = get_next_test_folder(base_params['testing'], base_params['learning'])

        if base_params['learning'] and not base_params['testing']:
            merge_q_tables(run_dirs, abs_output_dir, n_wolves=base_params["initial_wolves"])
            clean_up_q_tables(run_dirs)

        if save:
            save_simulation_metadata(base_params, q_learning_params, output_dir=output_dir)

        # plotting
        window_size = 100
        plot_results(df, output_dir=output_dir, window_size=window_size)
        plot_reward(df, output_dir=output_dir, window_size=window_size)
        plot_sheep_eaten(df, output_dir=output_dir, window_size=window_size)
        plot_simulation_steps(df, output_dir=output_dir, window_size=window_size)
        plot_all_actions_in_one(df, output_dir=output_dir, window_size=window_size)
        plot_capture_median(df, output_dir=output_dir, window_size=window_size)





