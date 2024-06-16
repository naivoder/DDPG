import gymnasium as gym
import torch
import numpy as np
import imageio
from agent import DDPGAgent
from argparse import ArgumentParser
import os
from utils import save_animation

def generate_animation(env_name, checkpoint_dir='checkpoints'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make_vec(env_name, num_envs=1)  # Single environment for rendering
    agent = DDPGAgent(env.single_observation_space.shape, env.single_action_space.shape, tau=0.001)
    agent.to(device)
    
    # Load the latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'{env_name}_best.pth')
    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
    else:
        print(f"Checkpoint {checkpoint_path} not found.")
        return
    
    frames = []
    states = env.reset()
    states = torch.tensor(states, dtype=torch.float32).to(device)
    term, trunc = False, False
    while not term and not trunc:
        frames.append(env.render(mode="rgb_array"))
        actions = agent.choose_action(states)
        next_states, rewards, term, trunc, _ = env.step(actions)
        states = torch.tensor(next_states, dtype=torch.float32).to(device)
    
    save_animation(frames, f"environments/{env_name}.gif")
    print(f"Animation saved as environments/{env_name}.gif")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--env", required=True, help="Environment name from Gymnasium"
    )
    args = parser.parse_args()
    generate_animation(args.env)
