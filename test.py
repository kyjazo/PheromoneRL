from modelpz import WolfSheepEnv
import numpy as np

env = WolfSheepEnv()

observations = env.reset()
print("Initial Observations:", observations)

num_steps = 10

for step in range(num_steps):
    print(f"\nStep {step + 1}:")

    actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}

    observations, rewards, dones, infos = env.step(actions)

    print("Actions:", actions)
    print("Observations:", observations)
    print("Rewards:", rewards)
    print("Dones:", dones)
    print("Infos:", infos)

    if all(dones.values()):
        print("\nSimulation finished early: all agents are done.")
        break

env.close()