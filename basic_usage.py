from magent2.environments import battle_v4
from pettingzoo.utils import random_demo

env = battle_v4.env(render_mode='human')
random_demo(env, render=True, episodes=1)