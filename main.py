import mesa
import pandas as pd
import matplotlib.pyplot as plt
from model import WolfSheepModel

if __name__ == "__main__":
    params = {"width": 100, "height": 100, "initial_wolves": 20, "initial_sheep": 100}


    result = mesa.batch_run(
        WolfSheepModel,
        data_collection_period=10,
        parameters={},
        iterations=1000,
        display_progress=True,
        number_processes=None,
        max_steps=100
    )


    df = pd.DataFrame(result)


    #df.to_csv("simulation_results.csv", index=False)
    #print("\nDati salvati in 'simulation_results.csv'")

    # Calcolare la media dei dati per ogni iterazione
    if "Sheep_eaten" in df.columns and "Avg_step_per_sheep" in df.columns:
        df_mean = df.groupby("iteration").mean()

        plt.figure(figsize=(12, 5))

        # Grafico per Sheep Eaten (media sulle iterazioni)
        plt.subplot(1, 2, 1)
        plt.plot(df_mean.index, df_mean["Sheep_eaten"], label="Average Sheep Eaten", color='red')
        plt.xlabel("Iteration")
        plt.ylabel("Number of Sheep Eaten")
        plt.title("Average Sheep Eaten Over Time")
        plt.legend()

        # Grafico per Avg Step per Sheep (media sulle iterazioni)
        plt.subplot(1, 2, 2)
        plt.plot(df_mean.index, df_mean["Avg_step_per_sheep"], label="Avg Steps per Sheep", color='blue')
        plt.xlabel("Iteration")
        plt.ylabel("Average Steps per Sheep")
        plt.title("Average Steps per Sheep Over Time")
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("Dati agent_reporters mancanti nella simulazione.")



#import time
#import psutil
#import mesa
#from model import WolfSheepModel
#
#def run_simulation(width, height, initial_wolves, initial_sheep, steps):
#    start_time = time.time()
#    process = psutil.Process()
#
#    params = {
#        "width": width,
#        "height": height,
#        "initial_wolves": initial_wolves,
#        "initial_sheep": initial_sheep
#    }
#
#    result = mesa.batch_run(
#        WolfSheepModel,
#        data_collection_period=1,
#        parameters=params,
#        iterations=10,
#        max_steps=steps,
#        display_progress=True
#    )
#
#    end_time = time.time()
#    execution_time = end_time - start_time
#    cpu_usage = process.cpu_percent(interval=1)
#    ram_usage = process.memory_info().rss / (1024 * 1024)  # in MB
#
#    return execution_time, cpu_usage, ram_usage
#
#
#scenarios = [
#    {"width": 20, "height": 20, "initial_wolves": 5, "initial_sheep": 20, "steps": 100},
#    {"width": 50, "height": 50, "initial_wolves": 10, "initial_sheep": 50, "steps": 500},
#    {"width": 100, "height": 100, "initial_wolves": 20, "initial_sheep": 100, "steps": 1000},
#]
#
#
#results = []
#for scenario in scenarios:
#    print(f"Running scenario: {scenario}")
#    execution_time, cpu_usage, ram_usage = run_simulation(
#        scenario["width"],
#        scenario["height"],
#        scenario["initial_wolves"],
#        scenario["initial_sheep"],
#        scenario["steps"]
#    )
#    results.append({
#        "scenario": scenario,
#        "execution_time": execution_time,
#        "cpu_usage": cpu_usage,
#        "ram_usage": ram_usage
#    })
#
#
#for result in results:
#    print(f"Scenario: {result['scenario']}")
#    print(f"Execution Time: {result['execution_time']} seconds")
#    print(f"CPU Usage: {result['cpu_usage']}%")
#    print(f"RAM Usage: {result['ram_usage']} MB")
#    print("-" * 40)