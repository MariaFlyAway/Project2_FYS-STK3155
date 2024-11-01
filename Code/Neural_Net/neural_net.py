import autograd.numpy as np 
from sklearn import datasets
from sklearn.metrics import accuracy_score
from autograd import grad
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

np.random.seed(2024)

# Defining activation functions
def identity_func(z):
    return z

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

def mean_squared_error_loss(predict, target):
    return np.mean((predict - target) ** 2)


# calculating the accuracy
def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)

# one-hot encoding the labels
def one_hot_encoder(input, labels):
    targets = np.zeros((len(input), labels))  
    for i, t in enumerate(input):
        targets[i, t] = 1
    
    return targets


class NeuralNet(ClassifierMixin, RegressorMixin, BaseEstimator):
    """
    Class that implements a neural net with multiple hidden layers of variable size 
    and uses Autograd to perform backpropagation. Inherits from ClassifierMixin, RegressorMixin 
    and BaseEstimator to create a custom estimator that is compatible with the scikit-learn API.
    
    Parameters
    ----------
    n_features: int
        Number of features of the input dataset. Default is 1.
    n_hidden: list
        List with size of each hidden layer. Last element of n_hidden is the number of outputs/classes. Default is [1], a single output.
    activations: list
        Activation function for each layer. Default is 'sigmoid'.
    cost_func: Callable
        Function used to calculate the loss. Default is cross_entropy.
    epochs: int
        Number of passes through the dataset. Default is 1000.
    batch_size: int
        Size of the mini-batches. Default is 16.
    epsilon: float
        Learning rate. Default is 0.01.
    """
    def __init__(self, 
            n_features=1,
            n_hidden=[1], 
            activations=['sigmoid'],
            loss_fn='cross_entropy',
            epochs=100,
            batch_size=16,
            epsilon=0.01,):

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.activations = activations
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsilon = epsilon

        # defining the cost and activation functions given the input strings
        self.activation_funcs = self._get_act_func(self.activations)
        self.loss_func= self._get_loss_func(self.loss_fn)
        
        
    def _get_act_func(self, activation_funcs):
        """
        Creates a list of the activation funcs.

        Parameters
        ----------
        activation_funcs: list
            list of names of activation functions

        Returns
        ----------
        activations: list
            list of activation functions
        """
        act_func_dict = {'sigmoid': sigmoid, 
                         'relu': ReLU, 
                         'softmax': softmax,
                         'identity': identity_func}

        activations = []
        for name in activation_funcs:
            activations.append(act_func_dict[name])

        return activations


    def _get_loss_func(self, loss_type):
        """
        Returns the loss function based on the provided loss type.

        Parameters
        ----------
        loss_type: str
            The type of loss function to use (e.g., 'cross_entropy', 'mse')
        
        Returns
        ----------
        loss_fn: Callable
            Loss function.
        """
        loss_funcs = {'cross_entropy': cross_entropy,
                      'mse': mean_squared_error_loss}
        
        return loss_funcs[loss_type]

    def _create_layers_batch(self):
        """
        Creates the layers of the neural net by 
        initializing the weights and biases.

        Returns
        ----------
        layers: list of tuples
            List where each element is a tuple containing the weight matrix and bias vector for a layer.
        """
        layers = []

        i_size = self.n_features
        for layer_output_size in self.n_hidden:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size
        return layers
    

    def cost(self, X, layers, y):
        predict = self.forwardpropagation(X, layers)
        return self.loss_func(predict, y)
    

    def forwardpropagation(self, X, layers):
        """
        Performs forward propagation through the network by updating
        the weights and biases.

        Parameters
        ----------
        X: np.ndarray
            Input data.
        
        Returns
        ----------
        a: np.ndarray
            Output from a given layer.
        """
        a = X
        for (W, b), activation in zip(layers, self.activation_funcs):
            z = np.dot(a, W) + b
            a = activation(z)
        return a
    
    
    def gradient_descent(self, X, y):
        """
        Performs gradient descent to update weights and biases.

        Parameters
        ----------
        X: np.ndarray
            Input data.
        y: np.ndarray
            Target labels.
        """
        layers_grad = self.gradient_func(X, self.layers, y)

        for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
            W -= self.epsilon * W_g
            b -= self.epsilon * b_g


    def fit(self, X, y):
        """
        Fits the neural network to the training data using 
        stochastic gradient descent.
        Prints the performance of the network for every 100th epoch.

        Parameters
        ----------
        X: np.ndarray
            Input data.
        y: np.ndarray
            Target labels.
        
        Returns
        ----------
        self: NeuralNet
            Fitted neural network.
        """
        self.classes_ = np.unique(y)
        self.loss_func = self._get_loss_func(self.loss_fn)
        self.layers = self._create_layers_batch()

        self.indices = np.arange(X.shape[0])
        indices = np.random.permutation(self.indices)
        self.gradient_func = grad(self.cost, 1) 
        batch_size = self.batch_size
        for i in range(self.epochs):
            for start in range(0, X.shape[0], batch_size):
                batch_indices = indices[start : start+batch_size]
                Xi, yi = X[batch_indices], y[batch_indices]

                self.gradient_descent(Xi, yi)
            
            # Print accuracy every 100 epochs
            if i % 100 == 0:  
                predictions = self.predict(X)
                if self.loss_fn == 'cross_entropy':
                    acc = accuracy_score(np.argmax(y, axis=1), predictions)
                    print(f"Epoch {i}: Accuracy = {acc}")
                else:
                    print(f"Epoch {i}: MSE = {mean_squared_error_loss(y, predictions)}")

        return self


    def predict(self, X):
        """
        Predicts class labels for the samples in X.

        Parameters
        ----------
        X: np.ndarray
            Input data.
        
        Returns
        ----------
        predictions: np.ndarray
            Predicted class labels.
        """
        probabilities = self.forwardpropagation(X, self.layers)
        if self.loss_fn == 'cross_entropy':
            return np.argmax(probabilities, axis=-1)
        else:
            return probabilities
        
    def predict_proba(self, X):
        """
        Predicts class probabilities for the samples in X.

        Parameters
        ----------
        X: np.ndarray
            Input data.
        
        Returns
        ----------
        probabilities: np.ndarray
            Predicted class probabilities.
        """
        return self.forwardpropagation(X, self.layers)
    
    def score(self, X, y):
        """
        Computes the accuracy to score 
        the performance of the neural net.

        Parameters
        ----------
        X: np.ndarray
            Input data.
        y: np.ndarray
            Target labels.
        
        Returns
        ----------
        score: float
            Accuracy score.
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
                    

if __name__ == "__main__":
    iris = datasets.load_iris()

    inputs = iris.data
    
    X_train, X_test, y_train, y_test = train_test_split(inputs, iris.target, test_size=0.2, random_state=3)
    
    network_input_size = 4
    layer_output_sizes = [8, 3]
    activations = ['sigmoid', 'softmax']

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', NeuralNet(network_input_size, layer_output_sizes, activations, epochs=100))
    ])
    pipeline.fit(X_train, one_hot_encoder(y_train, 3))

    predictions_train = pipeline.predict(X_train)
    predictions_test = pipeline.predict(X_test)

    print(f'Train accuracy: {pipeline.score(X_train, y_train)}')
    print(f'Test accuracy: {pipeline.score(X_test, y_test)}')