from custom_environment import env
from demo import demo

#1: up-left
#2: up
#3: up-right
#5: left
#6: idle
#7: right
#8: right (1 cell empty)
#9: bottom-left
#10:bottom
#11:bottom-right

def random_policy(env, agent, state): 
    return env.action_space(agent).sample()

demo(env(render_mode = 'human'), random_policy, render=True, episodes=1)