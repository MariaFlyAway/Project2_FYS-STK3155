import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from sklearn import datasets
from sklearn.metrics import accuracy_score
from autograd import grad
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from sklearn.pipeline import Pipeline

np.random.seed(2024)

# Defining some activation functions
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
    Class that implements a neural net with multiple hidden layers of variable size.
    
    Parameters
    ----------
        n_hidden (list): List with size of each hidden layer. Last element of n_hidden is the number of outputs/classes.
        activations (list): Activation function for each layer.
        cost_func (Callable): Function used to calculate the loss. Default is cross_entropy.
        epochs (int): Number of passes through the dataset. Default is 1000.
        batch_size (int): Size of the mini-batches. Default is 100.
        epsilon (float): Learning rate. Default is 0.001.
    """
    def __init__(self, 
            n_features=1,
            n_hidden=[1], 
            activations=[identity_func],
            loss_fn='cross_entropy',
            epochs=1,
            batch_size=100,
            epsilon=0.001):

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.activations = activations
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.activation_funcs = self._get_act_func(self.activations)
        self.loss_func= self._get_loss_func(self.loss_fn)
        
        
    def _get_act_func(self, activation_funcs):
        """
        Creates lists of the activation funcs and the derivatives of the activation funcs.
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
        loss_funcs = {'cross_entropy': cross_entropy,
                      'mse': mean_squared_error_loss}
        
        return loss_funcs[loss_type]

    def _create_layers_batch(self):
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
    
    def gradient_descent(self, X, y):
        layers_grad = self.gradient_func(X, self.layers, y)

        for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
            W -= self.epsilon * W_g
            b -= self.epsilon * b_g

    def forwardpropagation(self, X, layers):
        a = X
        for (W, b), activation in zip(layers, self.activation_funcs):
            z = np.dot(a, W) + b
            a = activation(z)
        return a

    def fit(self, X, y):
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
        probabilities = self.forwardpropagation(X, self.layers)
        if self.loss_fn == 'cross_entropy':
            return np.argmax(probabilities, axis=-1)
        else:
            return probabilities
        
    def predict_proba(self, X):
        return self.forwardpropagation(X, self.layers)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
                    

if __name__ == "__main__":
    iris = datasets.load_iris()

    inputs = iris.data
    
    X_train, X_test, y_train, y_test = train_test_split(inputs, iris.target, test_size=0.2, random_state=3)
    
    network_input_size = 4
    layer_output_sizes = [8, 3]
    activations = ['sigmoid', 'softmax']

    nn = NeuralNet(network_input_size, layer_output_sizes, activations, epochs=100)
    nn.fit(X_train, one_hot_encoder(y_train, 3))

    predictions_train = nn.predict(X_train)
    predictions_test = nn.predict(X_test)
    y_train_one_hot = one_hot_encoder(y_train, 3)
    y_test_one_hot = one_hot_encoder(y_test, 3)
    print(f'Train accuracy: {accuracy_score(y_train, predictions_train)}')
    print(f'Test accuracy: {accuracy_score(y_test, predictions_test)}')

    print(nn.score(X_test, y_test))


    # k_folds = KFold(n_splits=10)

    # pipeline = Pipeline([
    #     ('model', NeuralNet(network_input_size, layer_output_sizes, activations, batch_size=10, epochs=100))
    # ])
    # param_grid = {
    #     'model__epsilon': np.logspace(-4, -1, 4),
    # }

    # grid_search = GridSearchCV(estimator=pipeline,
    #                 param_grid=param_grid,
    #                 scoring='accuracy',
    #                 cv=k_folds,
    #                 verbose=1,
    #                 n_jobs=1)
    # gs = grid_search.fit(X_train, y_train)
    # print(gs.best_score_)
    # print(gs.best_params_)