import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import BaseEstimator, ClassifierMixin

np.random.seed(2024)

# Defining activation functions
def identity_func(z):
    return z

def ReLU(z):
    return np.where(z > 0, z, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Correct axis
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def softmax_vec(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

# defining derivatives of activation functions
def identity_der(z):
    return np.ones_like(z)  # Correct 1 to ones_like

def sigmoid_der(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def softmax_der(z):
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

def cross_entropy(predict, target):
    epsilon = 1e-12
    predict = np.clip(predict, epsilon, 1. - epsilon)
    return np.sum(-target * np.log(predict)) / predict.shape[0]

def mean_squared_error_loss(predict, target):
    return np.mean((predict - target) ** 2)

def mse_der(predict, target):
    return 2 * (predict - target) / predict.size

def cross_ent_der(predict, target):
    return -target / predict

def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)
    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)

def one_hot_encoder(input, labels):
    targets = np.zeros((len(input), labels))
    for i, t in enumerate(input):
        targets[i, t] = 1
    return targets

class NeuralNet(ClassifierMixin, BaseEstimator):
    def __init__(self,
                 n_features=1,
                 n_hidden=[1],
                 activations=['identity'],
                 loss_fn='cross_entropy',
                 epochs=1000,
                 batch_size=10,
                 epsilon=0.001):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.activations = activations
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.activation_funcs, self.activation_ders = self._get_act_func(self.activations)
        self.loss_func, self.loss_func_der = self._get_loss_func(self.loss_fn)

    def _get_act_func(self, activation_funcs):
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
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size
        return layers

    def gradient_descent(self, X, y):
        layers_grad = self.backpropagation(X, y)

        for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
            W -= self.epsilon * W_g
            b -= self.epsilon * b_g

    def backpropagation(self, X, y):
        layer_inputs, zs, predict = self.forwardpropagation(X)

        layer_grads = [() for layer in self.layers]

        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]

            if i == len(self.layers) - 1:
                dC_da = self.loss_func_der(predict, y)
            else:
                W_next, _ = self.layers[i + 1]
                dC_da = np.dot(dC_dz, W_next.T)

            if i == len(self.layers) - 1 and self.activation_funcs[i] == softmax:
                dC_dz = np.einsum('ij,ijk->ik', dC_da, softmax_der(z))
            else:
                dC_dz = dC_da * activation_der(z)

            dC_dW = np.dot(layer_input.T, dC_dz)
            dC_db = np.sum(dC_dz, axis=0)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def forwardpropagation(self, X):
        layer_inputs = []
        zs = []
        a = X
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = np.dot(a, W) + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.layers = self._create_layers_batch()
        y_one_hot = one_hot_encoder(y, len(self.classes_))

        self.indices = np.arange(X.shape[0])
        indices = np.random.permutation(self.indices)
        batch_size = self.batch_size
        for i in range(self.epochs):
            for start in range(0, X.shape[0], batch_size):
                batch_indices = indices[start: start + batch_size]
                Xi, yi = X[batch_indices], y_one_hot[batch_indices]

                self.gradient_descent(Xi, yi)

            # Print accuracy every 100 epochs
            if i % 100 == 0:
                predictions = self.predict(X)
                if self.loss_fn == 'cross_entropy':
                    acc = accuracy_score(y, predictions)
                    print(f"Epoch {i}: Accuracy = {acc}")
                else:
                    print(f"Epoch {i}: MSE = {mean_squared_error_loss(y, predictions)}")

        return self

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
        return accuracy_score(y, predictions)

if __name__ == "__main__":
    iris = datasets.load_iris()
    inputs = iris.data

    X_train, X_test, y_train, y_test = train_test_split(inputs, iris.target, test_size=0.2, random_state=3)

    network_input_size = 4
    layer_output_sizes = [8, 3]
    activations = ['sigmoid', 'softmax']

    nn = NeuralNet(network_input_size, layer_output_sizes, activations, loss_fn='cross_entropy')
    nn.fit(X_train, y_train)

    predictions_train = nn.predict(X_train)
    predictions_test = nn.predict(X_test)

    y_train_one_hot = one_hot_encoder(y_train, 3)
    y_test_one_hot = one_hot_encoder(y_test, 3)

    print(f'Train accuracy: {accuracy_score(y_train, predictions_train)}')
    print(f'Test accuracy: {accuracy_score(y_test, predictions_test)}')
    print(nn.score(X_test, y_test))
