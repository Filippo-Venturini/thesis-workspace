from custom_environment import env
from demo import demo

#ACTION SPACE: 0up 1sx 2idle 3dx 4down

def random_policy(env, agent, state): 
    return env.action_space(agent).sample()

custom_env = env(render_mode = 'human')
custom_env.reset()

demo(env(render_mode = 'human'), random_policy, render=True, episodes=1)