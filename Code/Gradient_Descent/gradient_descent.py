import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


# Cost function gradients
def MSE_OLS(X, y, theta):
    return 2/len(y) * (X.T @ (X @ theta - y))

def MSE_Ridge(X, y, theta, llambda):
    return 2/len(y) * X.T @ (X @ (theta) - y) + 2*llambda*theta


class GradientDescent(BaseEstimator, RegressorMixin):       # Inheritance adds compatibility with scikit-learn framework
    """
    Class that implements Gradient Descent with or without momentum.
    
    Attributes
    ----------
        epsilon (float): Learning rate.
        epochs (int, optional): Maximal number of passes through the dataset. Default is 1_000.
        tol (float, optional): Tolerance for stopping criteria. Default is 1e-6.
        momentum (float, optional): Momentum term for gradient update. Default is 0.
        cost_gradient (touple, optional): Callable cost function gradient with a list of its parameters. Default is OLS MSE.
    """
    def __init__(self, epsilon=0.01, epochs=1_000, tol=1e-6, momentum=0.0, cost_gradient=None):
        self.epsilon = epsilon
        self.epochs = epochs
        self.tol = tol
        self.momentum = momentum
        if cost_gradient is None:
            self.cost_gradient = (MSE_OLS, [])
        else:
            self.cost_gradient = cost_gradient


    def fit(self, X, y):
        """
        Fits the model using gradient descent.

        Args:
            X (np.ndarray): The input feature matrix.
            Y (np.ndarray): The output/target vector.

        Returns:
            tuple: A tuple containing the optimized parameters and the number of iterations performed.
        """
        self.n, p = X.shape
        tol = self.tol
        self.change = 2*tol # ~0
        self.theta = .1*np.random.randn(p) # Initial guess, range ~[-1,1] (perhaps very bad guess for unscaled data)
        self.indices = np.arange(self.n) # For SGD

        epoch = 0
        epochs = self.epochs
        while epoch < epochs and np.mean(np.abs(self.change)) > tol: # perhaps we should also add option to specify different stopping criteria?
            self._step(X,y)
            epoch += 1
        return self.theta, epoch


    def _step(self, X, y):
        cost_gradient, params = self.cost_gradient
        self.gradient = cost_gradient(X, y, self.theta, *params) 
        self.change = self._advance()
        self.theta -= self.change


    def _advance(self): # burde kanskje hete "_change", siden den finner self.change. Så kan _step hete _advance, f.eks.
        """
        Advances one step in the gradient descent algorithm.

        Returns:
            np.ndarray: The change in parameters to be subtracted from the current parameters.
        """
        return self.epsilon*self.gradient + self.momentum*self.change
    

    def predict(self, X):
        """
        Predicts the values of X with the calculated parameters theta.

        Returns:
            np.ndarray: The predicted values for input X.
        """
        return X @ self.theta


class StochasticGD(GradientDescent):
    """
    Class for implementing Stochastic Gradient Descent.

    Attributes
    ----------
        epochs (int): Number of passes through the dataset. Default is 1_000.
        batch_size (int, optional): Size of the mini-batches. Default is 16.
    """
    def __init__(self, epsilon=0.01, epochs=1000, tol=1e-6, momentum=0.0, cost_gradient=None, batch_size=16):
        super().__init__(epsilon, epochs, tol, momentum, cost_gradient)
        self.batch_size = batch_size

    def _step(self, X, y):
        indices = np.random.permutation(self.indices)
        batch_size = self.batch_size
        for start in range(0, self.n, batch_size):
            batch_indices = indices[start : start+batch_size]
            Xi, yi = X[batch_indices], y[batch_indices]

            cost_gradient, params = self.cost_gradient
            self.gradient = cost_gradient(Xi, yi, self.theta, *params)
            self.change = self._advance()
            self.theta -= self.change


class AdaGradGD(StochasticGD):
    """
    Implements the AdaGrad optimization algorithm.

    Attributes:
        delta (float): Small value to prevent division by zero in the update rule.
        r (float): Accumulated sum of squared gradients.
    """
    def __init__(self, 
                 epsilon=0.001, 
                 epochs=1_000, 
                 tol=1e-6, 
                 momentum=0.0,
                 cost_gradient=None,
                 batch_size=16,
                 delta=1e-7):
        super().__init__(epsilon, epochs, tol, momentum, cost_gradient, batch_size)
        self.delta = delta
        self.r = 0

    def _advance(self):
        self.r += self.gradient * self.gradient
        return self.epsilon/(self.delta + np.sqrt(self.r)) * self.gradient


class RMSPropGD(StochasticGD):
    """
    Implements the RMSProp optimization algorithm.

    Attributes:
        rho (float): Decay rate for the moving average of squared gradients.
        delta (float): Small value to prevent division by zero in the update rule.
        r (float): Moving average of squared gradients.
    """
    def __init__(self, 
                 epsilon=0.001, 
                 epochs=1_000, 
                 tol=1e-6, 
                 momentum=0.0,
                 cost_gradient=None, 
                 batch_size=16,
                 rho=0.9,
                 delta=1e-6):
        super().__init__(epsilon, epochs, tol, momentum, cost_gradient, batch_size)
        self.rho = rho
        self.delta = delta
        self.r = 0

    def _advance(self):
        self.r = self.rho * self.r + (1 - self.rho) * (self.gradient * self.gradient)
        return self.epsilon*self.gradient / np.sqrt(self.r + self.delta)
        

class ADAMGD(StochasticGD):
    """
    Implements the ADAM optimization algorithm.

    Attributes:
        rho1 (float): Exponential decay rate for the first moment estimates.
        rho2 (float): Exponential decay rate for the second moment estimates.
        delta (float): Small value to prevent division by zero in the update rule.
        r (float): Weighted average of squared gradients.
        s (float): Weighted average of gradients.
        t (int): Time step, used for bias correction.
    """
    def __init__(self, 
                 epsilon=0.001, 
                 epochs=1_000, 
                 tol=1e-6, 
                 momentum=0.0, 
                 cost_gradient=None,
                 batch_size=16,
                 rho1=0.9,
                 rho2 = 0.999,
                 delta=1e-8):
        super().__init__(epsilon, epochs, tol, momentum, cost_gradient, batch_size)
        self.rho1 = rho1
        self.rho2 = rho2
        self.delta = delta
        self.r = 0.0
        self.s = 0.0
        self.t = 0

    def _advance(self):
        self.t += 1
        self.s = self.rho1*self.s + (1 - self.rho1) * self.gradient
        self.r = self.rho2*self.r + (1 - self.rho2) * (self.gradient * self.gradient)

        s_hat = self.s/(1 - self.rho1**self.t)
        r_hat = self.r/(1 - self.rho2**self.t)

        return self.epsilon * s_hat/np.sqrt(r_hat) + self.delta

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error

    np.random.seed(3)

    n = 100
    x = 2*np.linspace(-1,1,n)
    y = 1 + 2*x + 3*x**2 #+ np.random.randn(n)

    p = 2
    X = np.c_[*(x**i for i in range(p+1))]

    ### Dette gir ikke mening nå når vi avslutter ved en gitt toleranse; da er jo alle like gode MSE-wise. 
    ### Men blir veldig nyttig når vi skal tilpasse ekte data / nevrale nett og kanskje ikke alltid konvergerer i tide!
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # param_grid_gd = {
    #     'epsilon': [1e-3, 1e-2, 1e-1],
    #     'momentum': [0.0, 0.5, 0.9],
    #     'epochs': [500, 1000, 2000],
    #     'batch_size': [16, 32, 64]
    # }

    # grid_search_gd = GridSearchCV(estimator=ADAMGD(cost_func=(MSE_Ridge, [0.001])), param_grid=param_grid_gd, cv=3, scoring='neg_mean_squared_error')
    # grid_search_gd.fit(X_train, y_train)

    # # Best parameters and score
    # print("Best parameters for GradientDescent:", grid_search_gd.best_params_)
    # print("Best score for GradientDescent:", grid_search_gd.best_score_)

    # # Predictions using the best model
    # best_gd_model = grid_search_gd.best_estimator_
    # y_pred_gd = best_gd_model.predict(X_test)
    
    # print("MSE for best GradientDescent model:", mean_squared_error(y_test, y_pred_gd))