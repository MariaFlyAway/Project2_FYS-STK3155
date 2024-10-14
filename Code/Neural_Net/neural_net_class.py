import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from sklearn import datasets
from sklearn.metrics import accuracy_score
from autograd import grad

np.random.seed(2024)

# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

# Defining the function used for calculating the loss
def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

class NeuralNet:
    def __init__(self, 
            X_data,
            y_data,
            n_hidden, 
            n_classes, 
            activations,
            cost_func=cross_entropy,
            epochs=1000,
            batch_size=100,
            epsilon=0.001):

        self.X_data_full = X_data
        self.y_data_full = y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.activations = activations

        self.cost_func = cost_func

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.epsilon = epsilon

        self.layers = self._create_layers_batch()

    def _create_layers_batch(self):
        layers = []

        i_size = self.n_features
        for layer_output_size in self.n_hidden:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size
        return layers
    
    def feed_forward_out(self, X, layers):
        a = X
        for (W, b), activation in zip(layers, self.activations):
            z = np.dot(a, W) + b
            a = activation(z)
        return a

    def forwardpropagation(self, layers):
        a = self.X_data_full
        for (W, b), activation in zip(layers, self.activations):
            z = np.dot(a, W) + b
            a = activation(z)
        return a

    def cost(self, layers):
        predict = self.forwardpropagation(layers)
        return cross_entropy(predict, self.y_data_full)
    
    def backpropagation(self):
        layers_grad = self.gradient_func(self.layers)

        for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
            W -= self.epsilon * W_g
            b -= self.epsilon * b_g

    def predict(self, X):
        probabilities = self.feed_forward_out(X, self.layers)
        return np.argmax(probabilities, axis=1)
    
    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X, self.layers)
        return probabilities

    def train_network(self):
        self.gradient_func = grad(self.cost, 0) 
        for i in range(self.epochs):
            self.forwardpropagation(self.layers)
            self.backpropagation()

            if i % 10 == 0:  # Print accuracy every 10 epochs
                predictions = self.forwardpropagation(self.layers)
                acc = accuracy(predictions, self.y_data_full)
                print(f"Epoch {i}: Accuracy = {acc}")

if __name__ == "__main__":
    iris = datasets.load_iris()

    inputs = iris.data

    # Since each prediction is a vector with a score for each of the three types of flowers,
    # we need to make each target a vector with a 1 for the correct flower and a 0 for the others.
    targets = np.zeros((len(iris.data), 3))
    for i, t in enumerate(iris.target):
        targets[i, t] = 1


    def accuracy(predictions, targets):
        one_hot_predictions = np.zeros(predictions.shape)

        for i, prediction in enumerate(predictions):
            one_hot_predictions[i, np.argmax(prediction)] = 1
        return accuracy_score(one_hot_predictions, targets)
    
    network_input_size = 4
    layer_output_sizes = [8, 3]
    activations = [sigmoid, softmax]
    nn = NeuralNet(inputs, targets, layer_output_sizes, layer_output_sizes[-1], activations)

    nn.train_network()

    predictions = nn.predict_probabilities(inputs)

    print(accuracy(predictions, targets))