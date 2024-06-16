import gymnasium as gym
import torch
from agent import DDPGAgent
from argparse import ArgumentParser
from utils import save_animation

def generate_animation(env_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name, render_mode="rgb_array")
    agent = DDPGAgent(env_name, env.observation_space.shape, env.action_space.shape, tau=0.001)
    agent.to(device)
    
    agent.load_checkpoints()
    
    frames = []
    states = env.reset()
    states = torch.tensor(states, dtype=torch.float32).to(device)
    term, trunc = False, False
    while not term and not trunc:
        frames.append(env.render())
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
