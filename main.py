import mesa
import pandas as pd
import matplotlib.pyplot as plt
from model import WolfSheepModel


def analyze_simulation(df):
    print("\n=== Performance Metrics ===")
    # Mostra le pecore mangiate per ogni iterazione
    sheep_per_iteration = df.groupby('iteration')['Sheep_eaten'].sum()
    print("\nSheep eaten per iteration:")
    print(sheep_per_iteration.to_string())


def plot_results(df):
    # Preparazione dati
    reward_stats = df.groupby('iteration')['Reward'].agg(['mean', 'std']).reset_index()
    sheep_eaten = df.groupby('iteration')['Sheep_eaten'].sum().reset_index()

    # Creazione figura con dimensioni adeguate
    plt.figure(figsize=(14, 6))

    # --- PRIMO GRAFICO: Reward Media con Deviazione Standard ---
    plt.subplot(1, 2, 1)

    # Linea della media con stile migliorato
    main_line = plt.plot(reward_stats['iteration'], reward_stats['mean'],
                         color='#1f77b4', linewidth=2.5, label='Reward Media',
                         marker='o', markersize=5, markeredgecolor='darkblue')

    # Area deviazione standard con trasparenza ottimale
    std_area = plt.fill_between(reward_stats['iteration'],
                                reward_stats['mean'] - reward_stats['std'],
                                reward_stats['mean'] + reward_stats['std'],
                                color='#1f77b4', alpha=0.15,
                                label='Deviazione Standard')

    plt.title('Evoluzione della Reward Media', fontsize=14, pad=15)
    plt.xlabel('Numero Simulazione', fontsize=12)
    plt.ylabel('Valore Reward', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10, framealpha=0.9)

    # Calcolo automatico dei limiti basato sui dati
    y_min = min((reward_stats['mean'] - reward_stats['std']).min(), reward_stats['mean'].min()) * 1.1
    y_max = max((reward_stats['mean'] + reward_stats['std']).max(), reward_stats['mean'].max()) * 1.1
    plt.ylim(y_min, y_max)

    # --- SECONDO GRAFICO: Pecore Mangiate per Simulazione (ora a linee) ---
    plt.subplot(1, 2, 2)

    # Linea con stile moderno
    line = plt.plot(sheep_eaten['iteration'], sheep_eaten['Sheep_eaten'],
                   color='#ff7f0e', linewidth=2.5, label='Pecore Mangiate',
                   marker='s', markersize=5, markeredgecolor='#d62728')

    plt.title('Andamento Pecore Mangiate', fontsize=14, pad=15)
    plt.xlabel('Numero Simulazione', fontsize=12)
    plt.ylabel('Numero di Pecore', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10, framealpha=0.9)

    # Calcolo automatico dei limiti con margine del 10%
    plt.ylim(0, sheep_eaten['Sheep_eaten'].max() * 1.1)

    # Ottimizzazione finale del layout
    plt.tight_layout(pad=3)
    plt.show()

if __name__ == "__main__":
    params = {"width": 20, "height": 20, "initial_wolves": 5, "initial_sheep": 20, "q_table_file": "q_table.json", "max_steps": 100}

    result = mesa.batch_run(
        WolfSheepModel,
        data_collection_period=100,
        parameters=params,
        iterations=100,
        display_progress=True,
        number_processes=1
    )

    df = pd.DataFrame(result)
    print(df)

    # Metodo 1 - Elimina righe dove Sheep_eaten è NaN
    df = df.dropna(subset=['Sheep_eaten'])
    print(df)
    if "Steps" in df.columns:
        print("\nMedia sheep eaten per iterazione:")
        print(df.groupby("iteration")["Sheep_eaten"].mean())
    else:
        print("Colonna 'Steps' non trovata nei dati della simulazione.")

    analyze_simulation(df)
    plot_results(df)
