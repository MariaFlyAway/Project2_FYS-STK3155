import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from sklearn import datasets
from sklearn.metrics import accuracy_score
from autograd import grad
from sklearn.model_selection import train_test_split

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


# calculating the accuracy
def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)

class NeuralNet:
    """
    Class that implements a neural net with multiple hidden layers of variable size.
    
    Attributes
    ----------
        X_data (np.ndarray): Input data.
        y_data (np.ndarray): Class labels.
        n_hidden (list): List with size of each hidden layer. Last element of n_hidden is the number of outputs/classes.
        activations (list): Activation function for each layer.
        cost_func (Callable): Function used to calculate the loss. Default is cross_entropy.
        epochs (int): Number of passes through the dataset. Default is 1000.
        batch_size (int): Size of the mini-batches. Default is 100.
        epsilon (float): Learning rate. Default is 0.001.
    """
    def __init__(self, 
            X_data,
            y_data,
            n_hidden, 
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
        self.n_classes = n_hidden[-1]
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
        a = self.X_data
        for (W, b), activation in zip(layers, self.activations):
            z = np.dot(a, W) + b
            a = activation(z)
        return a

    def cost(self, layers):
        predict = self.forwardpropagation(layers)
        return cross_entropy(predict, self.y_data)
    
    def gradient_descent(self):
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
        data_indices = np.arange(self.n_inputs)
        self.gradient_func = grad(self.cost, 0) 
        for i in range(self.epochs):
            for iter in range(self.iterations):
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training
                self.X_data, self.y_data = self.X_data_full[chosen_datapoints], self.y_data_full[chosen_datapoints]

                self.gradient_descent()
            
            if i % 100 == 0:  # Print accuracy every 10 epochs
                predictions = self.feed_forward_out(self.X_data_full, self.layers)
                acc = accuracy(predictions, self.y_data_full)
                print(f"Epoch {i}: Accuracy = {acc}")

if __name__ == "__main__":
    iris = datasets.load_iris()

    inputs = iris.data

    def one_hot_encoder(input):
        targets = np.zeros((len(input), 3))     # temporary hardcoding
        for i, t in enumerate(input):
            targets[i, t] = 1

        return targets
    
    X_train, X_test, y_train, y_test = train_test_split(inputs, iris.target, test_size=0.001, random_state=3)
    
    network_input_size = 4
    layer_output_sizes = [8, 3]
    activations = [sigmoid, softmax]

    nn = NeuralNet(X_train, one_hot_encoder(y_train), layer_output_sizes, activations)
    nn.train_network()
    #predictions = nn.predict_probabilities(X_test)
    #y_test_one_hot = one_hot_encoder(y_test)
    predictions_train = nn.predict_probabilities(X_train)
    y_train_one_hot = one_hot_encoder(y_train)
    print(f'Train accuracy: {accuracy(predictions_train, y_train_one_hot)}')
    #print(f'Test accuracy: {accuracy(predictions, y_test_one_hot)}')