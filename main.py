import gymnasium as gym
import numpy as np
from agent import DDPGAgent
from utils import plot_running_avg, save_animation
import pandas as pd
import warnings
from argparse import ArgumentParser
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

environments = [
    "BipedalWalker-v3",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Humanoid-v4",
    "LunarLanderContinuous-v2",
]


def run_ddpg(env_name, n_games=1000):
    env = gym.make(env_name)
    agent = DDPGAgent(
        env_name, env.observation_space.shape, env.action_space.shape, tau=0.001
    )

    best_score = env.reward_range[0]
    history = []
    metrics = []

    for i in range(n_games):
        state, _ = env.reset()
        agent.action_noise.reset()

        term, trunc, score = False, False, 0
        while not term and not trunc:
            action = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, term or trunc)
            agent.learn()

            score += reward
            state = next_state

        history.append(score)
        avg_score = np.mean(history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints(i + 1, score)

        metrics.append(
            {
                "episode": i + 1,
                "score": score,
                "average_score": avg_score,
                "best_score": best_score,
            }
        )

        print(
            f"[{env_name} Episode {i + 1:04}/{n_games}]    Score = {score:7.4f}    Average = {avg_score:7.4f}",
            end="\r",
        )

    plot_running_avg(history, env_name)
    df = pd.DataFrame(metrics)
    df.to_csv(f"metrics/{env_name}_metrics.csv", index=False)

    frames = []
    state, _ = env.reset()
    term, trunc = False, False
    while not term and not trunc:
        frames.append(env.render())
        action = agent.choose_action(state)
        next_state, reward, term, trunc, _ = env.step(action)
        state = next_state

    save_animation(frames, f"environments/{env_name}.gif")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", default=None, help="Environment name from Gymnasium"
    )
    args = parser.parse_args()

    for fname in ["metrics", "environments", "weights"]:
        if not os.path.exists(fname):
            os.makedirs(fname)

    if args.env:
        run_ddpg(args.env)
    else:
        for env_name in environments:
            run_ddpg(env_name)
