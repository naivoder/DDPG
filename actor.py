import torch
import numpy as np


class ActorNetwork(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        action_space,
        h1_size=400,
        h2_size=300,
        lr=1e-4,
        chkpt_path="weights/actor.pt",
    ):
        super(ActorNetwork, self).__init__()
        self.input_shape = input_shape
        self.action_space = action_space
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.lr = lr
        self.chkpt_path = chkpt_path

        self.h1_layer = torch.nn.Linear(*self.input_shape, self.h1_size)
        self.h2_layer = torch.nn.Linear(self.h1_size, self.h2_size)

        # use layer norm b/c it isn't affected by batch size
        # batch norm also fails to copy running avg to target networks
        self.ln1 = torch.nn.LayerNorm(self.h1_size)
        self.ln2 = torch.nn.LayerNorm(self.h2_size)

        self.out_layer = torch.nn.Linear(self.h2_size, self.action_space.shape[0])

        self.init_weights()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.action_min = torch.tensor(self.action_space.low, device=self.device)
        self.action_max = torch.tensor(self.action_space.high, device=self.device)
        self.action_range = self.action_max - self.action_min

    def init_weights(self):
        f1 = 1.0 / np.sqrt(self.h1_layer.weight.data.size()[0])
        self.h1_layer.weight.data.uniform_(-f1, f1)
        self.h1_layer.bias.data.uniform_(-f1, f1)

        f2 = 1.0 / np.sqrt(self.h2_layer.weight.data.size()[0])
        self.h2_layer.weight.data.uniform_(-f2, f2)
        self.h2_layer.bias.data.uniform_(-f2, f2)

        fout = 3e-3
        self.out_layer.weight.data.uniform_(-fout, fout)
        self.out_layer.bias.data.uniform_(-fout, fout)

    def forward(self, state):
        # doing layer norm prior to relu so it accounts for negative values
        state = self.h1_layer(state)
        state = torch.nn.functional.relu(self.ln1(state))

        state = self.h2_layer(state)
        state = torch.nn.functional.relu(self.ln2(state))

        output = torch.nn.functional.tanh(self.out_layer(state))
        scaled_output = self.action_min + (output + 1.0) * 0.5 * self.action_range
        return scaled_output

    def save_checkpoint(self, epoch, loss):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            self.chkpt_path,
        )

    def load_checkpoint(self):
        chkpt = torch.load(self.chkpt_path)
        self.load_state_dict(chkpt["model_state_dict"])
        self.optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        epoch = chkpt["epoch"]
        loss = chkpt["loss"]
        return epoch, loss
