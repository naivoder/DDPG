import torch
import numpy as np


class ActorNetwork(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
        h1_size=400,
        h2_size=300,
        lr=1e-4,
        chkpt_path="weights/actor.pt",
    ):
        super(ActorNetwork, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
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

        self.out_layer = torch.nn.Linear(self.h2_size, *self.n_actions)

        self.init_weights()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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

        return torch.nn.functional.tanh(self.out_layer(state))

    def save_checkpoint(self, epoch, loss):
        # torch.save(self.state_dict(), self.chkpt_path)
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
        # self.load_state_dict(torch.load(self.chkpt_path))
        chkpt = torch.load(self.chkpt_path)
        self.load_state_dict(chkpt["model_state_dict"])
        self.optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        epoch = chkpt["epoch"]
        loss = chkpt["loss"]
        return epoch, loss
