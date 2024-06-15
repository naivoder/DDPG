import gymnasium as gym
import numpy as np
from agent import DDPGAgent
from utils import plot_running_avg, save_animation
import pandas as pd
import warnings
from argparse import ArgumentParser

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--env", default="LunarLanderContinuous-v2")
    args = parser.parse_args()

    N_GAMES = 1000

    env = gym.make(args.env)
    agent = DDPGAgent(env.observation_space.shape, env.action_space.shape, tau=0.001)

    best_score = env.reward_range[0]
    history = list()
    metrics = []

    for i in range(N_GAMES):
        state, info = env.reset()
        agent.action_noise.reset()

        term, trunc, score = False, False, 0
        while not term and not trunc:
            action = agent.choose_action(state)
            next_state, reward, term, trunc, info = env.step(action)
            # done = True if term or trunc else False

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
            f"[Episode {i + 1:04}/{N_GAMES}]\tScore = {score:.4f}\tAverage = {avg_score:4f}",
            end="\r",
        )

    plot_running_avg(history, args.env)
    df = pd.DataFrame(metrics)
    df.to_csv("results/{args.env}_metrics.csv", index=False)

    frames = []
    state, info = env.reset()
    term, trunc = False, False
    while not term and not trunc:
        frames.append(env.render(mode="rgb_array"))
        action = agent.choose_action(state)
        next_state, reward, term, trunc, info = env.step(action)
        state = next_state

    save_animation(frames, f"environments/{args.env}.gif")
