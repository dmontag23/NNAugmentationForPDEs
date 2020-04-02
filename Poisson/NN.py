import torch
import torch.nn as nn
import numpy as np

class Network(nn.Module):

    gradients = []

    def __init__(self):
        super(Network, self).__init__()
        self.inputSize = 1
        self.hidden1Size = 3
        self.outputSize = 1

        # creates a layer with weights W and biases b in the form y=Wx+b
        self.hidden1 = nn.Linear(self.inputSize, self.hidden1Size).double()
        self.output = nn.Linear(self.hidden1Size, self.outputSize).double()

        # Define sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.output(x)
        self.compute_gradients(x)
        return x

    def update_parameters(self, p):
        params = np.split(p, [self.hidden1Size, 2 * self.hidden1Size, 3* self.hidden1Size])
        params[0] = params[0].reshape((3,1))
        params[2] = params[2].reshape((1,3))
        torch_params = [torch.from_numpy(i) for i in params]
        nn_params = list(self.parameters())
        for i in range(len(nn_params)):
            nn_params[i].data = torch_params[i]

    def compute_gradients(self, output):
        self.gradients = [[] for i in range(self.number_of_params())]
        output_vector = output.detach().numpy().flatten()
        output_size = len(output_vector)
        for i in range(output_size):
            self.zero_grad()
            y_for_jacobian_mvp = torch.zeros(output_size, 1)
            y_for_jacobian_mvp[i] = 1.0
            output.backward(y_for_jacobian_mvp, retain_graph=True)
            gradients_at_xi = np.concatenate([param.grad.numpy().flatten() for param in self.parameters()])
            for j in range(len(gradients_at_xi)):
                self.gradients[j].append(gradients_at_xi[j])

    def number_of_params(self):
        return sum([np.prod(p.size()) for p in self.parameters()])

    def print_params(self):
        for param in self.parameters():
            print(param)