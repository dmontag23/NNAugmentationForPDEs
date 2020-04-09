import numpy as np
import torch
import torch.nn as nn

class Network(nn.Module):

    def __init__(self, input_size, hidden_layer_size):
        super(Network, self).__init__()
        self.inputSize = input_size
        self.hidden1Size = hidden_layer_size
        self.outputSize = 1

        # creates a layer with weights W and biases b in the form y=Wx+b
        self.hidden1 = nn.Linear(self.inputSize, self.hidden1Size).double()
        self.output = nn.Linear(self.hidden1Size, self.outputSize).double()

        # Define sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    # given an input, run the nn to produce and output
    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x

    # calculate the jvp where the Jacobian is dOutput/dParameters and the vector to multiply it by is "gradient"
    def calculate_gradients(self, gradient, output):
        gradient_torch = torch.from_numpy(gradient.reshape(len(gradient), 1))
        self.zero_grad()
        output.backward(gradient_torch)

    # given a np array of parameters, update the parameters of the nn
    def update_parameters(self, p):
        params = np.split(p, [self.inputSize * self.hidden1Size, self.inputSize * self.hidden1Size + self.hidden1Size,
                              self.inputSize * self.hidden1Size + 2 * self.hidden1Size])
        params[0] = params[0].reshape((self.hidden1Size,self.inputSize))
        params[2] = params[2].reshape((1,self.hidden1Size))
        torch_params = [torch.from_numpy(i) for i in params]
        nn_params = list(self.parameters())
        for i in range(len(nn_params)):
            nn_params[i].data = torch_params[i]

    # return the number of parameters in the nn
    def number_of_params(self):
        return sum([np.prod(p.size()) for p in self.parameters()])

    # print the tensors of all the parameters in the nn
    def print_params(self):
        for param in self.parameters():
            print(param)