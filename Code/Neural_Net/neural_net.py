import autograd.numpy as np

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

# Defining cost functions
def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))


class NeuralNetwork:
    """
    Class that implements a neural net with multiple hidden layers of variable size.
    
    Attributes
    ----------
        X_data (np.ndarray): Input data.
        y_data (np.ndarray): Class labels.
        n_hidden (list): List with size of each hidden layer.
        n_classes (int): Number of class labels, output size.
        activations (list): Activation function for each layer.
        epochs (int): Number of passes through the dataset. Default is 50.
        batch_size (int): Size of the mini-batches. Default is 100.
        epsilon (float): Learning rate. Default is 0.1.
        llambda (float): Regularization parameter. Default is no regularization.
    """
    def __init__(
            self, 
            X_data,
            y_data,
            n_hidden, 
            n_classes, 
            activations,
            epochs=50,
            batch_size=100,
            epsilon=0.1,
            llambda=0.0):

        self.X_data = X_data
        self.y_data = y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.activations = activations

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.epsilon = epsilon
        self.llambda = llambda

        self._create_layers_batch(self.n_features, n_hidden)


    def _create_layers_batch(self, network_input_size, layer_output_sizes):
        """
        Initializes the matrix of weights and biases

        Returns:
            list: Weights and biases for each layer.
        """
        layers = []

        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size
        return layers
    

    def _activation(self, activation_func):
        """
        Currently a placeholder, outputs the sigmoid function, 
        want to implement multiple forms of activation functions
        
        Returns:
            The desired activation function
        """
        return activation_func


    def forwardpropagation(self, inputs, layers, activations):
        a = inputs
        for (W, b), activation in zip(layers, activations):
            z = np.dot(a, W) + b
            a = activation(z)
        return a
    
    def cost(self, input, layers, activations, target):
        predict = self.forwardpropagation(input, layers, activations)
        return cross_entropy(predict, target)

    def backpropagation(self, X, y):
        """Updates the weights in the backwards direction"""
        a_h, a_out = self.forwardpropagation(X)

        # output layer
        delta_out = a_out - y
        #print(0.5*((a_h-y)**2))

        # hidden layer
        delta_h = np.matmul(delta_out, self.weight_out.T) * a_h * (1 - a_h)

        # gradients for the output layer
        output_weights_gradient = np.matmul(a_h.T, delta_out)
        output_bias_gradient = np.sum(delta_out, axis=0)
        # gradient for the hidden layer
        hidden_weights_gradient = np.matmul(X.T, delta_h)
        hidden_bias_gradient = np.sum(delta_h, axis=0)

        return (output_weights_gradient, 
                output_bias_gradient, 
                hidden_weights_gradient, 
                hidden_bias_gradient)
    
    def train_network(
        self, inputs, layers, activations, targets, learning_rate=0.001, epochs=100
    ):
        for i in range(epochs):
            layers_grad = gradient_func(inputs, layers, activations, targets)

            for (W, b), (W_g, b_g) in zip(layers, layers_grad):
                W -= learning_rate * W_g
                b -= learning_rate * b_g

        predict = self.forwardpropagation(input, layers, activations)
        return predict