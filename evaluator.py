import time

import numpy as np
import torch


def evaluate(
    env,
    brain_name,
    action_size,
    actor,
    device,
    n_episodes=100,
):
    """Evaluation method.
    Params
    ======
        env: Unity Environment
        brain_name (str): name of the brain
        action_size(int): action size
        actor: the trained actor
        device: device cpu or gpu
        n_episodes (int): maximum number of training episodes
    """
    scores_agent_1 = []
    scores_agent_2 = []
    for _ in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        actor.reset_parameters()
        score_agent_1 = 0
        score_agent_2 = 0
        while True:
            states = torch.from_numpy(states).float().to(device)
            actions = np.zeros((len(env_info.agents), action_size))
            actor.eval()
            with torch.no_grad():
                for agent_num, state in enumerate(states):
                    action = actor(state).cpu().data.numpy()
                    actions[agent_num, :] = action
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score_agent_1 += rewards[0]
            score_agent_2 += rewards[1]
            states = next_states
            if np.any(dones):
                break
            time.sleep(0.05)
        scores_agent_1.append(score_agent_1)
        scores_agent_2.append(score_agent_2)

    max_score = max(np.mean(scores_agent_1), np.mean(scores_agent_2))
    print("\nAfter 100 episodes!\tAverage Score: {:.2f}".format(max_score))
