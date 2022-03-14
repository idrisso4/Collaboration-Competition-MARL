from collections import deque

import numpy as np
import torch

from agent import Agent


def train(env, brain_name: str, agent_config: dict, n_episodes=2000):
    """Train Method.
    Params
    ======
        env: Unity Environment
        brain_name (str): name of the brain
        agent_config (dict): config of the agent
        n_episodes (int): maximum number of training episodes
    """

    scores_deque = deque(maxlen=100)
    scores = []
    avg_score_list = []
    agent = Agent(**agent_config)

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(len(env_info.agents))
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            score += rewards
            states = next_states
            if np.any(dones):
                break

        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        avg_score = np.mean(scores_deque)
        avg_score_list.append(avg_score)

        if i_episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}".format(
                    i_episode, avg_score, np.mean(score)
                ),
                end="\n",
            )

        if np.mean(scores_deque) > 0.5:
            print(f"Enviroment solved in episode={i_episode} avg_score={avg_score:.2f}")

            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")

            return avg_score_list

    return avg_score_list
