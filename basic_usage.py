from magent2.environments import battle_v4
from pettingzoo.utils import random_demo

#env = battle_v4.env(render_mode='human')
#random_demo(env, render=True, episodes=1)

import numpy as np

# Supponiamo di avere un array numpy
array = np.array([1, 3, 2, 3, 1])

# Trova il valore massimo nell'array
max_value = np.max(array)

# Trova tutti gli indici dove il valore è uguale al valore massimo
max_indices = np.where(array == max_value)[0]

print(max_indices)
