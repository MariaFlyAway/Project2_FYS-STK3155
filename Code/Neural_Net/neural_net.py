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


class NeuralNet:
    def __init__(self, n_features, n_hidden, n_classes, random_state):
        """Initializes the weights and the bias"""
        # self.n_features = n_features
        # self.n_hidden = n_hidden
        # self.n_classes = n_classes
        
        # hidden layer
        rng = np.random.RandomState(random_state)
        
        self.weight_h = rng.normal(
            loc=0.0, scale=0.1, size=(n_hidden, n_features))
        self.bias_h = np.zeros(n_hidden)
        
        # output
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(n_classes, n_hidden))
        self.bias_out = np.zeros(n_classes)

    def _activation(self, z):
        """Currently a placeholder, outputs the sigmoid function, 
        want to implement multiple forms of activation functions"""
        return 1.0/(1.0+np.exp(-z))


    def forwardpropagation(self, X):
        """Updates the weights in the forward direcion"""
        # hidden layer
        z_h = np.matmul(X, self.weight_h) + self.bias_h
        a_h = self._activation(z_h)

        # output layer
        z_out = np.matmul(a_h, self.weight_out) + self.bias_out
        a_out = self._activation(z_out)
        return a_h, a_out

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