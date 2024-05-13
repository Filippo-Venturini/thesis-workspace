from pettingzoo.utils import random_demo
from custom_environment import env

random_demo(env(render_mode = 'human'), render=True, episodes=1)