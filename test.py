from modelpz import WolfSheepEnv
from pettingzoo.test import parallel_api_test

env = WolfSheepEnv()
parallel_api_test(env, num_cycles=1000)









#env = WolfSheepEnv()
#observations, infos = env.reset()
#
#while True:
#    actions = {agent: env.action_space.sample() for agent in env.agents}
#    observations, rewards, dones, infos = env.step(actions)
#    env.render()
#
#    if all(dones.values()):
#        break