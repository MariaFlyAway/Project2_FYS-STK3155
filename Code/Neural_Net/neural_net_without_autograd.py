import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from sklearn import datasets
from sklearn.metrics import accuracy_score
from autograd import jacobian
from autograd import elementwise_grad as egrad
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from sklearn.pipeline import Pipeline

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
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


# defining derivatives of activation functions
def identity_der(z):
    return 1

def sigmoid_der(z):
    return sigmoid(z)*(1-sigmoid(z))

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

# Defining the function used for calculating the loss
def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

def cross_entropy_reg(predict, target, llambda, layers):
    return np.sum(-target * np.log(predict)) + llambda*np.sum(layers**2)        # will not work as is, just a note

def mean_squared_error_loss(predict, target):
    return np.mean((predict - target) ** 2)

def mse_with_l2_penalty(predict, target, layers, llambda):
    weight_list = []
    for layer in layers:
        W, b = layer
        weight_list.append(W)

    mse_loss = np.mean((predict - target) ** 2)
    l2_penalty = np.sum([np.sum(W ** 2) for W in weight_list]) * llambda
    total_loss = mse_loss + l2_penalty
    
    return total_loss


# defining the derivatives of the loss functions
def mse_der(predict, target):
    return 2/predict.size*(predict - target)

def cross_ent_der(predict, target):
    return -target / predict


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

def softmax_der_alt(z):
    s = softmax(z)
    jacobian_m = np.zeros((z.shape[0], z.shape[1], z.shape[1]))
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            for k in range(z.shape[1]):
                if j == k:
                    jacobian_m[i][j][k] = s[i][j] * (1 - s[i][j])
                else:
                    jacobian_m[i][j][k] = -s[i][j] * s[i][k]
    return jacobian_m

def softmax_der(z):
    return egrad(softmax)(z)


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
            activations=['identity'],
            loss_fn='cross_entropy',
            epochs=1,
            batch_size=16,
            epsilon=0.01,
            llambda=0.0):

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.activations = activations
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.llambda = llambda

        self.activation_funcs, self.activation_ders = self._get_act_func(self.activations)
        self.loss_func, self.loss_func_der = self._get_loss_func(self.loss_fn)
        

    def _get_act_func(self, activation_funcs):
        """
        Creates lists of the activation funcs and the derivatives of the activation funcs.
        """
        act_func_dict = {'sigmoid': [sigmoid, sigmoid_der], 
                         'relu': [ReLU, ReLU_der], 
                         'softmax': [softmax, softmax_der],
                         'identity': [identity_func, identity_der]}

        activations = []
        activation_ders = []
        for name in activation_funcs:
            activations.append(act_func_dict[name][0])
            activation_ders.append(act_func_dict[name][1])

        return activations, activation_ders


    def _get_loss_func(self, loss_type):
        loss_funcs = {'cross_entropy': (cross_entropy, cross_ent_der),
                      'mse': (mean_squared_error_loss, mse_der)}
        
        loss, loss_der = loss_funcs[loss_type]
        
        return loss, loss_der


    def _create_layers_batch(self):
        layers = []

        i_size = self.n_features
        for layer_output_size in self.n_hidden:
            W = np.random.randn(layer_output_size, i_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size
        return layers
    

    def forwardpropagation(self, X):
        layer_inputs = []
        zs = []
        a = X
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = np.matmul(a, W.T) + b
            a = activation_func(z)
            
            zs.append(z)

        return layer_inputs, zs, a


    def backpropagation(self, X, y):
        layer_inputs, zs, predict = self.forwardpropagation(X)

        layer_grads = [() for layer in self.layers]

        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]

            if i == len(self.layers) - 1:
                dC_da = self.loss_func_der(predict, y)
            else:
                (W, b) = self.layers[i+1]
                dC_da = np.dot(dC_dz, W)

            prev_W, prev_b = self.layers[i]
            reg = self.llambda * prev_W

            # if self.activation_funcs[i] == softmax:
            #     dC_dz = np.einsum('ij,ijk->ik', dC_da, softmax_der_alt(z))
            # else:
            dC_dz = dC_da * activation_der(z)

            dC_dW = np.dot(dC_dz.T, layer_input) + reg
            dC_db = np.sum(dC_dz, axis=0)

            layer_grads[i] = (dC_dW, dC_db)
    
        return layer_grads


    def fit(self, X, y):
        self.classes_ = np.unique(y)        # finds number of class labels
        self.layers = self._create_layers_batch()

        self.indices = np.arange(X.shape[0])
        batch_size = self.batch_size
        for i in range(self.epochs):
            indices = np.random.permutation(self.indices)
            for start in range(0, X.shape[0], batch_size):
                batch_indices = indices[start : start+batch_size]
                Xi, yi = X[batch_indices], y[batch_indices]

                self.gradient_descent(Xi, yi)
            
            #Print accuracy every 100 epochs
            if i % 100 == 0:  
                predictions = self.predict(X)
                if self.loss_fn == 'cross_entropy':
                    acc = accuracy_score(np.argmax(y, axis=1), predictions)
                    print(f"Epoch {i}: Accuracy = {acc}")
                else:
                    print(f"Epoch {i}: MSE = {mean_squared_error_loss(y, predictions)}")

        return self
    

    def gradient_descent(self, X, y): # should be separated
        layers_grad = self.backpropagation(X, y)

        for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
            W -= self.epsilon * W_g
            b -= self.epsilon * b_g


    def predict(self, X):
        _, _, probabilities = self.forwardpropagation(X)
        if self.loss_fn == 'cross_entropy':
            return np.argmax(probabilities, axis=-1)
        else:
            return probabilities
        
    def predict_proba(self, X):
        _, _, probabilities = self.forwardpropagation(X)
        return probabilities
    
    def score(self, X, y):
        predictions = self.predict(X)
        if self.loss_fn == 'cross_entropy':
            score = accuracy_score(y, predictions)
        else:
            score = mean_squared_error_loss(y, predictions)
        return score
                    

if __name__ == "__main__":
    iris = datasets.load_iris()

    inputs = iris.data
    
    X_train, X_test, y_train, y_test = train_test_split(inputs, iris.target, test_size=0.2, random_state=3)
    
    network_input_size = 4
    layer_output_sizes = [8, 3]
    activations = ['sigmoid', 'sigmoid']

    nn = NeuralNet(network_input_size, 
                   layer_output_sizes, 
                   activations, 
                   loss_fn='cross_entropy', 
                   epsilon=0.1, 
                   epochs=100, 
                   batch_size=16)
    nn.fit(X_train, one_hot_encoder(y_train, 3))

    predictions_train = nn.predict(X_train)
    predictions_test = nn.predict(X_test)
    y_train_one_hot = one_hot_encoder(y_train, 3)
    y_test_one_hot = one_hot_encoder(y_test, 3)
    print(f'Train accuracy: {accuracy_score(y_train, predictions_train)}')
    print(f'Test accuracy: {accuracy_score(y_test, predictions_test)}')

    print(nn.score(X_test, y_test))


    k_folds = KFold(n_splits=10)

    pipeline = Pipeline([
        ('model', NeuralNet(network_input_size, layer_output_sizes, activations, loss_fn='cross_entropy', batch_size=10, epochs=100))
    ])
    param_grid = {
        'model__epsilon': np.logspace(-4, -1, 4),
    }

    grid_search = GridSearchCV(estimator=pipeline,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=k_folds,
                    verbose=1,
                    n_jobs=1)
    gs = grid_search.fit(X_train, one_hot_encoder(y_train, 3))
    print(-gs.best_score_)
    print(gs.best_params_)