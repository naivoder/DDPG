import torch
import numpy as np
from noise import OrnsteinUhlenbeckActionNoise
from actor import ActorNetwork
from critic import CriticNetwork
from memory import ReplayBuffer
from copy import deepcopy


class DDPGAgent(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        n_actions,
        tau,
        alpha=1e-4,
        beta=1e-3,
        gamma=0.99,
        batch_size=64,
        mem_size=int(1e6),
    ):
        super(DDPGAgent, self).__init__()
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(input_dims, n_actions, buffer_length=mem_size)
        self.action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(n_actions))

        self.actor = ActorNetwork(input_dims, n_actions, lr=self.alpha)
        self.target_actor = ActorNetwork(input_dims, n_actions, lr=self.alpha)

        self.critic = CriticNetwork(input_dims, n_actions, lr=self.beta)
        self.target_critic = CriticNetwork(input_dims, n_actions, lr=self.beta)

        self.update_network_parameters(tau=1)

    def choose_action(self, state):
        self.actor.eval()
        # state = torch.Tensor(np.array(state), dtype=torch.float).to(self.actor.device)
        state = torch.Tensor(np.array(state)).to(self.actor.device)
        mu = self.actor(state).to(self.actor.device)

        # add noise to deterministic output
        mu = mu + torch.tensor(self.action_noise(), dtype=torch.float).to(
            self.actor.device
        )
        self.actor.train()
        # return mu.cpu().detach().numpy()[0]
        return mu.cpu().detach().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

    def save_checkpoints(self, epoch, loss):
        self.actor.save_checkpoint(epoch, loss)
        self.target_actor.save_checkpoint(epoch, loss)
        self.critic.save_checkpoint(epoch, loss)
        self.target_critic.save_checkpoint(epoch, loss)

    def load_checkpoints(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.replay_buffer.mem_counter < self.batch_size:
            return

        states, actions, rewards, next_states, done = self.replay_buffer.sample(
            self.batch_size
        )
        states = torch.Tensor(states).to(self.actor.device)
        actions = torch.Tensor(actions).to(self.actor.device)
        next_states = torch.Tensor(next_states).to(self.actor.device)
        rewards = torch.Tensor(rewards).to(self.actor.device)
        done = torch.Tensor(done).to(self.actor.device).to(torch.bool)

        target_actions = self.target_actor(next_states)
        target_critic_values = self.target_critic(next_states, target_actions)
        critic_values = self.critic(states, actions)

        # set target critic value to zero for terminal states
        target_critic_values[done] = 0.0  # fix dim issue
        target_critic_values = target_critic_values.view(-1)

        target = rewards + self.gamma * target_critic_values
        target = target.view(self.batch_size, 1)  # add batch dim

        self.critic.optimizer.zero_grad()
        critic_loss = torch.nn.functional.mse_loss(target, critic_values)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states))
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return actor_loss, critic_loss

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = dict(self.actor.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        for name in actor_params:
            actor_params[name] = (
                tau * actor_params[name].clone()
                + (1 - tau) * target_actor_params[name].clone()
            )
        self.target_actor.load_state_dict(actor_params)

        critic_params = dict(self.critic.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())
        for name in critic_params:
            critic_params[name] = (
                tau * critic_params[name].clone()
                + (1 - tau) * target_critic_params[name].clone()
            )
        self.target_critic.load_state_dict(critic_params)

        # To use batch norm instead of layer norm:
        # self.target_actor.load_state_dict(actor_params, strict=False)
        # self.critic_actor.load_state_dict(critic_params)
