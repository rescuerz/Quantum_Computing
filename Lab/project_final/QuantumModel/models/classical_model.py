import torch
import torch.optim as optim
from torch.autograd import Variable
import pennylane as qml
import pennylane.numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
import warnings

from .base_model import BaseModel

class ClassicalModel(BaseModel):
    def __init__(self, output_shape=10, **hp):
        super().__init__(**hp)
        self.nb_hidden_neurons = self.hp.get("nb_hidden_neurons", 2)
        self.clayer_1 = torch.nn.Linear(8*8, 2*self.nb_hidden_neurons)
        self.clayer_2 = torch.nn.Linear(self.nb_hidden_neurons, self.nb_hidden_neurons)
        self.clayer_3 = torch.nn.Linear(self.nb_hidden_neurons, self.nb_hidden_neurons)
        self.clayer_4 = torch.nn.Linear(2*self.nb_hidden_neurons, output_shape)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.clayer_1(x)
        x_1, x_2 = torch.split(x, int(self.nb_hidden_neurons), dim=1)
        x_1 = self.clayer_2(x_1)
        x_2 = self.clayer_3(x_2)
        x = torch.cat([x_1, x_2], axis=1)
        x = self.clayer_4(x)
        return self.softmax(x)

class ClassicalClassifier(torch.nn.Module):
    def __init__(self, input_shape, output_shape, **hp):
        super(ClassicalClassifier, self).__init__()
        self.hp = hp
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nb_hidden_neurons = hp.get("nb_hidden_neurons", 1_000)
        self.linear_block = torch.nn.Sequential(
            torch.nn.Linear(self.nb_hidden_neurons, self.nb_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
        )
        self.seq = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(self.input_shape), self.nb_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            *[self.linear_block for _ in range(hp.get("nb_hidden_layer", 1))],
            torch.nn.Linear(self.nb_hidden_neurons, self.output_shape),
            torch.nn.Softmax(),
        )

    def forward(self, x):
        return self.seq(x)