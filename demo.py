import pdb
import random
import numpy as np
from pettingzoo.utils.env import AECEnv

def demo(env: AECEnv, policy, render: bool = True, episodes: int = 1) -> float:
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
                """ print(env)
                print(agent)
                print(obs[0][0][-2:]) """
                state = obs[0][0][-2:]
                action = policy(state)

            env.step(action)
        completed_episodes += 1

    if render:
        env.close()

    print("Average total reward", total_reward / episodes)

    return total_reward