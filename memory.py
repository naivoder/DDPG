import numpy as np


class ReplayBuffer:
    def __init__(self, input_shape, n_actions, buffer_length=1000):
        self.buffer_length = buffer_length
        self.mem_counter = 0
        self.state_memory = np.zeros((self.buffer_length, *input_shape))
        self.next_state_memory = np.zeros((self.buffer_length, *input_shape))
        self.action_memory = np.zeros((self.buffer_length, *n_actions))
        self.reward_memory = np.zeros((self.buffer_length))
        self.terminal_memory = np.zeros(
            self.buffer_length, dtype=bool
        )  # use as mask for critic

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.buffer_length  # clever...
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample(self, batch_size):
        mem_size = min(self.mem_counter, self.buffer_length)
        batch = np.random.choice(mem_size, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, next_states, dones
