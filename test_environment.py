import numpy as np
import random
from custom_environment import env
from pettingzoo.utils.env import AECEnv
import pdb


def random_demo(env: AECEnv, render: bool = True, episodes: int = 1) -> float:
    """Runs an env object with random actions."""
    total_reward = 0
    completed_episodes = 0

    while completed_episodes < episodes:
        env.reset()
        for agent in env.agent_iter():
            if render:
                env.render()

            obs, reward, termination, truncation, _ = env.last()
            total_reward += reward
            if termination or truncation:
                action = None
            elif isinstance(obs, dict) and "action_mask" in obs:
                action = random.choice(np.flatnonzero(obs["action_mask"]).tolist())
            else:
                action = env.action_space(agent).sample()
                print(env.state())
                
                pdb.set_trace()
            env.step(action)
        completed_episodes += 1

    if render:
        env.close()

    print("Average total reward", total_reward / episodes)

    return total_reward


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

random_demo(env(render_mode = 'human'), render=True, episodes=1)