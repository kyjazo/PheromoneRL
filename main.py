#import mesa
#from model import WolfSheepModel
#
#if __name__ == "__main__":
#    params = {"width": 20, "height": 20, "initial_wolves": 5, "initial_sheep": 20}
#
#    result = mesa.batch_run(WolfSheepModel, data_collection_period=1, parameters=params, iterations=1000, display_progress=True)
#
import time
import psutil
import mesa
from model import WolfSheepModel

def run_simulation(width, height, initial_wolves, initial_sheep, steps):
    start_time = time.time()
    process = psutil.Process()

    params = {
        "width": width,
        "height": height,
        "initial_wolves": initial_wolves,
        "initial_sheep": initial_sheep
    }

    result = mesa.batch_run(
        WolfSheepModel,
        data_collection_period=1,
        parameters=params,
        iterations=10,
        max_steps=steps,
        display_progress=True
    )

    end_time = time.time()
    execution_time = end_time - start_time
    cpu_usage = process.cpu_percent(interval=1)
    ram_usage = process.memory_info().rss / (1024 * 1024)  # in MB

    return execution_time, cpu_usage, ram_usage


scenarios = [
    {"width": 20, "height": 20, "initial_wolves": 5, "initial_sheep": 20, "steps": 100},
    {"width": 50, "height": 50, "initial_wolves": 10, "initial_sheep": 50, "steps": 500},
    {"width": 100, "height": 100, "initial_wolves": 20, "initial_sheep": 100, "steps": 1000},
]


results = []
for scenario in scenarios:
    print(f"Running scenario: {scenario}")
    execution_time, cpu_usage, ram_usage = run_simulation(
        scenario["width"],
        scenario["height"],
        scenario["initial_wolves"],
        scenario["initial_sheep"],
        scenario["steps"]
    )
    results.append({
        "scenario": scenario,
        "execution_time": execution_time,
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage
    })


for result in results:
    print(f"Scenario: {result['scenario']}")
    print(f"Execution Time: {result['execution_time']} seconds")
    print(f"CPU Usage: {result['cpu_usage']}%")
    print(f"RAM Usage: {result['ram_usage']} MB")
    print("-" * 40)