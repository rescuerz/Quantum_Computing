import torch
import torch.optim as optim
from torch.autograd import Variable
import pennylane as qml
import pennylane.numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
import warnings
import time
from .base_model import BaseModel
from pennylane.templates import RandomLayers
from scipy.ndimage import generic_filter

class QuantumConvolutionLayer(torch.nn.Module):
    def __init__(self, **hp):
        super().__init__()
        self.kernel_size = hp.get("kernel_size", (2, 2))
        self.nb_qubits = hp.get("nb_qubits", 2)
        dev = qml.device("default.qubit", wires=self.nb_qubits)

        # Random circuit parameters
        np.random.seed(0)
        rand_params = np.random.uniform(high=2 * np.pi, size=self.kernel_size)
        self.weights = torch.tensor(rand_params).float()

        get_circuit = lambda phi, layer_param: self.circuit(phi, layer_param)
        self.qnode = qml.QNode(get_circuit, dev, interface='torch')

        self.output_shape = None

    def circuit(self, phi=None, layer_params=None):
        # Encoding of 4 classical input values
        for j in range(self.nb_qubits):
            qml.RY(np.pi * phi[j], wires=j)

        # Random quantum circuit
        RandomLayers(layer_params, wires=list(range(self.nb_qubits)))

        # Measurement producing n classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(self.nb_qubits)]

    def forward(self, x):
        if self.output_shape is None:
            self.output_shape = (
                x.shape[0],
                x.shape[1] * self.nb_qubits,
                x.shape[2] // self.kernel_size[0],
                x.shape[3] // self.kernel_size[1]
            )
        output_shape = (x.shape[0], x.shape[1]*self.nb_qubits, x.shape[2] // self.kernel_size[0], x.shape[3] // self.kernel_size[1])
        out = torch.zeros(output_shape).to(x.device)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i] = self.convolve(x[i, j])
        return out

    def convolve(self, x):
        out = torch.zeros((self.nb_qubits, x.shape[0] // self.kernel_size[0], x.shape[1] // self.kernel_size[1])).to(x.device)
        for j in range(0, x.shape[0] - 1):
            for k in range(0, x.shape[1] - 1):
                q_results = self.q_filter(x[j:j + self.kernel_size[0], k:k + self.kernel_size[1]].flatten())
                for c in range(self.nb_qubits):
                    out[c, j // self.kernel_size[0], k // self.kernel_size[1]] = q_results[c]
        return out

    def q_filter(self, phi):
        return self.qnode(phi, self.weights)

class QuantumPseudoLinearLayer(torch.nn.Module):
    def __init__(self, **hp):
        super().__init__()
        self.nb_qubits = hp.get("nb_qubits", 2)
        self.n_layers = hp.get("n_layers", 6)
        dev = qml.device("default.qubit", wires=self.nb_qubits)
        get_circuit = lambda inputs, weights: self.circuit(inputs, weights)
        self.qnode = qml.QNode(get_circuit, dev, interface='torch')
        weight_shapes = {"weights": (self.n_layers, self.nb_qubits)}
        self.seq = qml.qnn.TorchLayer(self.qnode, weight_shapes)

    def circuit(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(self.nb_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(self.nb_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.nb_qubits)]

    def forward(self, x):
        return self.seq(x)

class QuantumClassifier(BaseModel):
    def __init__(self, input_shape, output_shape, **hp):
        super().__init__(**hp)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nb_qubits = hp.get("nb_qubits", 2)
        self.seq_0 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(self.input_shape), self.nb_qubits),
            *[QuantumPseudoLinearLayer(nb_qubits=self.nb_qubits, **hp)
              for _ in range(hp.get("nb_q_layer", 2))],
        )
        self.seq_1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(self.get_output_shape_seq_0()), self.output_shape),
            torch.nn.Softmax(),
        )

    def get_output_shape_seq_0(self):
        ones = torch.Tensor(np.ones((1, *self.input_shape))).float()
        return self.seq_0(ones).shape

    def forward(self, x):
        return self.seq_1(self.seq_0(x))