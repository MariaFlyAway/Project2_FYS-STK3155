import autograd.numpy as np
from autograd import grad

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
        cost_func (Callable): Function used to calculate the loss. Default is cross_entropy.
        epochs (int): Number of passes through the dataset. Default is 50.
        batch_size (int): Size of the mini-batches. Default is 100.
        epsilon (float): Learning rate. Default is 0.1.
    """
    def __init__(
            self, 
            X_data,
            y_data,
            n_hidden, 
            n_classes, 
            activations,
            cost_func=cross_entropy,
            epochs=50,
            batch_size=100,
            epsilon=0.1):

        self.X_data = X_data
        self.y_data = y_data

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
        """
        Initializes the matrix of weights and biases

        Returns:
            list: Weights and biases for each layer.
        """
        layers = []

        i_size = self.n_features
        for layer_output_size in self.n_hidden:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size
        return layers
    

    def cost(self, input, layers, activations, target):
        predict = self.forwardpropagation()
        return self.cost_func(predict, target)
    
    # def _activation(self, activation_func):
    #     """
    #     Currently a placeholder, outputs the sigmoid function, 
    #     want to implement multiple forms of activation functions
        
    #     Returns:
    #         The desired activation function
    #     """
    #     return activation_func


    def forwardpropagation(self):
        """
        ~Propagates through the neural net

        Returns:
            a (np.ndarray): The output of the layer
        """
        a = self.X_data
        for (W, b), activation in zip(self.layers, self.activations):
            z = np.dot(a, W) + b
            a = activation(z)
        return a
    
    def feed_forward_out(self, X):
        """
        ~Propagates through the neural net

        Returns:
            a (np.ndarray): The output of the layer
        """
        a = X
        for (W, b), activation in zip(self.layers, self.activations):
            z = np.dot(a, W) + b
            a = activation(z)
        return a
    
    def backpropagation(self):
        gradient_func = grad(self.cost, 1)      # Taking the gradient wrt. the second input to the cost function
        layers_grad = gradient_func(self.X_data, self.layers, self.activations, self.y_data)

        for (W, b), (W_g, b_g) in zip(self.layers, layers_grad):
            W -= self.epsilon * W_g
            b -= self.epsilon * b_g

    
    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities #np.argmax(probabilities, axis=1)
    
    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities
    
    def train_network(self):
        data_indices = np.arange(self.n_inputs)
        for i in range(self.epochs):
            for iter in range(self.iterations):
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training
                self.Xi, self.yi = self.X_data[chosen_datapoints], self.y_data[chosen_datapoints]

                self.forwardpropagation()
                self.backpropagation()


if __name__ == "__main__":
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier

    def one_hot_encoder(data):
        targets = np.zeros((len(data), 3))
        for i, t in enumerate(data):
            targets[i, t] = 1
        return targets


    def accuracy(predictions, targets):
        one_hot_predictions = np.zeros(predictions.shape)

        for i, prediction in enumerate(predictions):
            one_hot_predictions[i, np.argmax(prediction)] = 1
        return accuracy_score(one_hot_predictions, targets)

    iris = datasets.load_iris()

    _, ax = plt.subplots()
    scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
    ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
    _ = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
    )

    inputs = iris.data
    targets = one_hot_encoder(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=3)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    epochs = 500
    batch_size = 100
    network_input_size = 4
    layer_output_sizes = [8, 3]
    activations = [sigmoid, softmax]

    dnn = NeuralNetwork(inputs, targets, epochs=epochs, batch_size=batch_size,
                        n_hidden=layer_output_sizes, n_classes=layer_output_sizes[-1], activations=activations)
    dnn.train_network()
    test_predict = dnn.predict(X_test)

    # accuracy score from scikit library
    print("Accuracy score on test set: ", accuracy(test_predict, one_hot_encoder(y_test)))

    # equivalent in numpy
    # def accuracy_score_numpy(Y_test, Y_pred):
    #     return np.sum(Y_test == Y_pred) / len(Y_test)

    # print("Accuracy score on test set: ", accuracy_score_numpy(y_test, test_predict))


#     eta_vals = np.logspace(-5, 1, 7)
#     lmbd_vals = np.logspace(-5, 1, 7)

#     # store models for later use
#     DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
#     best_accuracy = 0
#     best_model = None

#     for i, eta in enumerate(eta_vals):
#         for j, lmbd in enumerate(lmbd_vals):
#             dnn = MLPClassifier(hidden_layer_sizes=[8,3],
#                                 alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
#             dnn.fit(X_train, y_train)
            
#             DNN_scikit[i][j] = dnn

#             # Calculate accuracy on the test set
#             accuracy = dnn.score(X_test, y_test)
            
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_model = dnn
            
#             print("Learning rate  = ", eta)
#             print("Lambda = ", lmbd)
#             print("Accuracy score on test set: ", dnn.score(X_test, y_test))
#             print()

# print(f"Best model: Learning rate = {best_model.learning_rate_init}, Lambda = {best_model.alpha}, Accuracy = {best_accuracy:.4f}")
