from custom_environment import env
from qlearning_agent import QLearningAgent
from demo import demo

# Usare l'ambiente Magent2
custom_env = env()
custom_env.reset()
QAgent = None
for agent in custom_env.agent_iter():
    custom_env.reset()
    QAgent = QLearningAgent(custom_env, agent)
    QAgent.train(10000)
    break

def policy(state):
    return QAgent.choose_action(state)

demo(env(render_mode = 'human'), QAgent.choose_action, render=True, episodes=1)