import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, n_inputs, bias=0., weights=None):
        self.b = bias
        if weights:
            self.ws = np.array(weights)
        else:
            self.ws = np.random.rand(n_inputs)
        self.input = None
        self.output = None

    def _f(self, x):
        return max(x * .1, x)

    def _f_derivative(self, x):
        return 0.1 if x < 0 else 1

    def __call__(self, xs):
        self.input = xs
        z = xs @ self.ws + self.b
        self.output = self._f(z)
        return self.output

def forward_pass(layers, input_data):
    activations = [input_data]
    for layer in layers:
        input_data = np.array([neuron(input_data) for neuron in layer])
        activations.append(input_data)
    return activations

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def backpropagate(layers, activations, y_true, learning_rate):
    y_pred = activations[-1]
    loss_deriv = compute_loss_derivative(y_true, y_pred)
    layer_errors = [None] * len(layers)

    for i in reversed(range(len(layers))):
        layer = layers[i]
        errors = []

        if i == len(layers) - 1:
            for j, neuron in enumerate(layer):
                error = loss_deriv[j]
                errors.append(error * neuron._f_derivative(neuron.input @ neuron.ws + neuron.b))
        else:
            next_layer = layers[i + 1]
            for j, neuron in enumerate(layer):
                error = sum(next_neuron.ws[j] * layer_errors[i + 1][k] for k, next_neuron in enumerate(next_layer))
                errors.append(error * neuron._f_derivative(neuron.input @ neuron.ws + neuron.b))

        layer_errors[i] = errors

        for j, neuron in enumerate(layer):
            input_to_neuron = activations[i]
            neuron.ws -= learning_rate * errors[j] * input_to_neuron
            neuron.b -= learning_rate * errors[j]

def train_network(layers, input_data, y_true, epochs, learning_rate):
    for epoch in range(epochs):
        activations = forward_pass(layers, input_data)
        backpropagate(layers, activations, y_true, learning_rate)
        if epoch % 100 == 0:
            loss = compute_loss(y_true, activations[-1])
            print(f"Epoch {epoch}, Loss: {loss}")

def visualize_network(layers):
    fig, ax = plt.subplots()

    n_inputs = len(layers[0][0].ws)
    n_hidden_neurons_1 = len(layers[0])
    n_hidden_neurons_2 = len(layers[1])
    n_outputs = len(layers[2])

    for i in range(n_inputs):
        ax.plot([0, 1], [i, i], 'r-', alpha=0.5)
        ax.plot(1, i, 'rs', markersize=10)
        ax.text(1.1, i, f"Input {i + 1}", ha='center', va='center')
    ax.text(-0.2, n_inputs / 2 - 0.5, 'input layer', fontsize=10, color='r', rotation=90, ha='center', va='center')

    for i in range(n_hidden_neurons_1):
        for j in range(n_inputs):
            ax.plot([1, 1.5], [j, i], 'b-', alpha=0.5)
        ax.plot(1.5, i, 'bo', markersize=10)
        ax.text(1.6, i, f"Hidden 1-{i + 1}", ha='center', va='center')
    ax.text(1.25, n_hidden_neurons_1 / 2 - 0.5, 'hidden layer 1', fontsize=10, color='b', rotation=90, ha='center', va='center')

    for i in range(n_hidden_neurons_2):
        for j in range(n_hidden_neurons_1):
            ax.plot([1.5, 2], [j, i], 'b-', alpha=0.5)
        ax.plot(2, i, 'bo', markersize=10)
        ax.text(2.1, i, f"Hidden 2-{i + 1}", ha='center', va='center')
    ax.text(1.75, n_hidden_neurons_2 / 2 - 0.5, 'hidden layer 2', fontsize=10, color='b', rotation=90, ha='center', va='center')

    for i in range(n_outputs):
        for j in range(n_hidden_neurons_2):
            ax.plot([2, 2.5], [j, i], 'b-', alpha=0.5)
        ax.plot(2.5, i, 'go', markersize=10)
        ax.text(2.6, i, f"Output {i + 1}", ha='center', va='center')
    ax.text(2.25, 0, 'output layer', fontsize=10, color='g', rotation=90, ha='center', va='center')

    ax.set_xlabel('Neurons')
    ax.set_ylabel('Layer')
    ax.set_title('Neural Network Structure')

    plt.tight_layout()
    plt.show()


n_inputs = 3
n_hidden_neurons_1 = 4
n_hidden_neurons_2 = 4
n_outputs = 1
learning_rate = 0.01
epochs = 1000


layers = [
    [Neuron(n_inputs) for _ in range(n_hidden_neurons_1)],
    [Neuron(n_hidden_neurons_1) for _ in range(n_hidden_neurons_2)],
    [Neuron(n_hidden_neurons_2, weights=[1.0] * n_hidden_neurons_2) for _ in range(n_outputs)],
]

np.random.seed(42)
input_data = np.random.rand(n_inputs)
y_true = np.array([0.5])

visualize_network(layers)


train_network(layers, input_data, y_true, epochs, learning_rate)
