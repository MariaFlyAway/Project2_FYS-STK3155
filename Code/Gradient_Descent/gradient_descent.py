import numpy as np
#from typing import Callable

np.random.seed(3)

class GradientDescent:
    """
    Class that implements Gradient Descent with and without momentum.
    
    Attributes
    ----------
    X : np.ndarray
        The input feature matrix.
    Y : np.ndarray
        The output/target vector.
    cost : Callable (default 0) 
            Cost function to be minimized (not implemented currently).
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, cost=0):
        self.X = X
        self.Y = Y
        self.cost = cost # IF cost function, we should use AutoGrad (not implemented)

    def fit(self, 
            derivative, 
            eta, 
            max_iter=1_000,
            tol=1e-6,
            momentum=0):
        """
        Fits the model using gradient descent.

        Args:
            derivative (callable): Derivative of the cost function w.r.t parameters.
            eta (float): Learning rate.
            max_iter (int, optional): Maximum number of iterations. Default is 1_000.
            tol (float, optional): Tolerance for stopping criteria. Default is 1e-6.
            momentum (float, optional): Momentum term for gradient update. Default is 0.

        Returns:
            tuple: A tuple containing the optimized parameters and the number of iterations performed.
        """
        
        self.eta = eta
        self.momentum = momentum

        X, Y = self.X, self.Y
        n, p = X.shape

        idx = 0
        self.change = 2*tol # Might be too radical
        theta = .1*np.random.randn(p) # Initial guess, range ~[-1,1] (perhaps very bad guess for unscaled data)

        while idx < max_iter and np.mean(np.abs(self.change)) > tol: # Could make entire in-loop as advance at cost of indices-matrix
            self.gradient = derivative(X, Y, theta)
            self.change = self._advance()
            theta -= self.change
            idx += 1
        return theta, idx

    def _advance(self):
        """
        Advances one step in the gradient descent algorithm.

        Returns:
            np.ndarray: The change in parameters to be subtracted from the current parameters.
        """
        return self.eta*self.gradient + self.change*self.momentum


class StochasticGD(GradientDescent):
    def fit(self, 
            derivative, 
            eta: float, 
            epochs=1_000,
            batch_size=16,
            tol=1e-6,
            momentum=0):
        """
        Fits the model using stochastic gradient descent.

        Args:
            derivative (callable): Derivative of the cost function w.r.t parameters.
            eta (float): Learning rate.
            epochs (int, optional): Maximum number of iterations. Default is 1_000.
            batch_size (int, optional): Size of the mini-batches. Default is 16.
            tol (float, optional): Tolerance for stopping criteria. Default is 1e-6.
            momentum (float, optional): Momentum term for gradient update. Default is 0.

        Returns:
            tuple: A tuple containing the optimized parameters and the number of iterations performed.
        """
        self.eta = eta
        self.momentum = momentum
        
        X, Y = self.X, self.Y 
        n, p = X.shape

        epoch = 0
        self.change = 2*tol
        theta = np.random.randn(p)

        indices = np.arange(n)
        while epoch < epochs and np.mean(np.abs(self.change)) > tol:
            np.random.shuffle(indices)

            for start in range(0,n,batch_size):
                batch_indices = indices[start:start+batch_size]
                Xi, Yi = X[batch_indices], Y[batch_indices]

                self.gradient = derivative(Xi, Yi, theta)
                self.change = self._advance()
                theta -= self.change
            epoch += 1

        return theta, epoch


class RMSPropGD(StochasticGD):
    def __init__(self, X: np.ndarray, Y: np.ndarray, cost=0):
        super().__init__(X, Y, cost)
        self.rho = 0.9
        self.delta = 1e-6
        self.epsilon = 1e-3
        self.r = 0

    def advance(self):
        self.r = self.rho * self.r + (1 - self.rho)*self.gradient @ self.gradient
        return self.eta*self.gradient / np.sqrt(self.r + self.delta)
        


class AdaGradGD(StochasticGD):
    def __init__(self, X: np.ndarray, Y: np.ndarray, cost=0):
        super().__init__(X, Y, cost)
        self.delta = 1e-7
        self.epsilon = 1e-3
        self.r = 0

    def advance(self):
        self.r = self.r + self.gradient @ self.gradient
        return self.epsilon/(self.delta + np.sqrt(self.r)) * self.gradient
    

class ADAMGD(StochasticGD):
    def __init__(self, X: np.ndarray, Y: np.ndarray, cost=0):
        super().__init__(X, Y, cost)
        self.epsilon = 0.001
        self.rho1 = 0.9
        self.rho2 = 0.999
        self.delta = 1e-2
        self.r = 0
        self.s = 0
        self.t = 0

    def advance(self):
        self.t = self.t + 1
        self.s = self.rho1*self.s + (1 - self.rho1) * self.gradient
        self.r = self.rho2*self.r + (1 - self.rho2) * self.gradient @ self.gradient

        s_hat = self.s/(1 - self.rho1**self.t)
        r_hat = self.r/(1 - self.rho2**self.t)

        return self.epsilon * s_hat/np.sqrt(r_hat) + self.delta



# Testing 
if __name__ == "__main__":
    def gradient_OLS(X, y, theta):
        n = X.shape[0]
        return (2.0/n)*X.T @ (X @ theta - y)

    # def gradient_Ridge(X, y, beta):
    #     llambda = 0.001
    #     n = X.shape[0]
    #     return 2.0/n*X.T @ (X@beta - y)+2*llambda*beta

    n = 100
    x = 2*np.linspace(-1,1,n)
    y = 1 + 2*x + 3*x**2 #+ np.random.randn(n)

    p = 2
    X = np.c_[*(x**i for i in range(p+1))]
    
    gd = GradientDescent(X, y)
    theta = gd.fit(gradient_OLS, 0.1, 1000, momentum=.5)

    sgd = StochasticGD(X, y)
    theta2 = sgd.fit(gradient_OLS, 0.1, 1000, 16, momentum=.5)

    rms = RMSPropGD(X, y)
    theta3 = rms.fit(gradient_OLS, 0.1, 1000, 16, momentum=0.5)

    ada = AdaGradGD(X, y)
    theta4 = ada.fit(gradient_OLS, 0.1, 1000, 16, momentum=0.5)

    adam = ADAMGD(X, y)
    theta5 = adam.fit(gradient_OLS, 0.1, 1000, 16, momentum=0.5)

    print(theta)
    print(theta2)
    print(theta3)
    print(theta4)
    print(theta5)
