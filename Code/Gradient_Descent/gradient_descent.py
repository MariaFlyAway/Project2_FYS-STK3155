import numpy as np
from typing import Callable

np.random.seed(3)

class GradientDescent:
    def __init__(self, X: np.ndarray, 
                 Y: np.ndarray, 
                 derivative, 
                 eta: float, 
                 max_iter: int,
                 momentum=0):
        self.X = X
        self.Y = Y
        self.derivative = derivative
        self.eta = eta
        self.max_iter = max_iter
        self.momentum = momentum

    def fit(self):
        n_iter = 0
        theta = 0.1*np.random.randn(self.X.shape[1])
        theta_new = np.zeros_like(theta)
        tol = 1e-9
        self.change = 0.0
        while np.mean(np.abs(theta - theta_new)) >= tol and n_iter < self.max_iter:
            self.gradient = self.derivative(self.X, self.Y, theta)
            self.change = self.advance()
            theta = theta_new.copy()
            theta_new -= self.change
            n_iter += 1

        return theta_new, n_iter

    def advance(self):
        return self.eta*self.gradient + self.change*self.momentum


class StochasticGD(GradientDescent):
    def __init__(self, X, Y, derivative, eta, max_iter, momentum=0, batch_size=16):
        """"""
        super().__init__(X, Y, derivative, eta, max_iter, momentum)
        self.batch_size = batch_size

    def fit(self):
        n_iter = 0
        n, p  = self.X.shape
        theta = 0.1*np.random.randn(p)
        theta_new = np.zeros_like(theta)
        tol = 1e-9
        self.change = 0.0
        batch_size = self.batch_size
        X, Y = self.X, self.Y

        indices = np.arange(n)
        while np.mean(np.abs(theta - theta_new)) >= tol and n_iter < self.max_iter:
            np.random.shuffle(indices)

            for start in range(0, n, batch_size):
                batch_indices = indices[start:start+batch_size]
                Xi, Yi = X[batch_indices], Y[batch_indices]

                self.gradient = self.derivative(Xi, Yi, theta)
                self.change = self.advance()
                theta = theta_new.copy()
                theta_new -= self.change
            n_iter += 1

        return theta_new, n_iter

class RMSPropGD(GradientDescent):
    def advance(self):
        delta = 1e-6
        epsilon = 1e-3
        rho = 0.9
        r = rho * r + (1 - rho)*self.gradient @ self.gradient
        return self.eta*self.gradient / np.sqrt(r + delta)

    def fit(self):
        super.fit()
        r = 0


class AdaGrad(GradientDescent):
    def fit(self):
        super.fit()
        self.r = 0
        self.epsilon = 1e-3
        self.delta = 1e-7

    def advance(self):
        self.r = self.r + self.gradient @ self.gradient
        return self.epsilon/(self.delta + np.sqrt(self.r)) * self.gradient
    

class ADAM(GradientDescent):
    def fit(self):
        super.fit()
        self.r = 0
        self.s = 0
        self.t = 0
        self.epsilon = 0.001
        self.rho1 = 0.9
        self.rho2 = 0.999
        self.delta = 1e-2

    def advance(self):
        self.t = self.t + 1
        self.s = self.rho1*self.s + (1 - self.rho1) * self.gradient
        self.r = self.rho2*self.r + (1 - self.rho2) * self.gradient @ self.gradient

        s_hat = self.s/(1 - self.rho1**self.t)
        r_hat = self.r/(1 - self.rho2**self.t)

        return self.epsilon * s_hat/np.sqrt(r_hat) + self.delta

def gradient_OLS(X, y, theta):
    n = X.shape[0]
    return (2.0/n)*X.T @ (X @ theta - y)

def gradient_Ridge(X, y, beta):
    llambda = 0.001
    n = X.shape[0]
    return 2.0/n*X.T @ (X@beta - y)+2*llambda*beta

def derivative(x):
    return x * 6.0 + 2.0

if __name__ == "__main__":
    n = 100
    x = np.linspace(0, 1, n)
    y = 3*x**2 + 2*x + 1 #+ np.random.randn(n,1)

    p = 2
    X = np.c_[*(x**i for i in range(p+1))]
    
    gd = GradientDescent(X, y, gradient_OLS, 0.001, 1000)
    theta = gd.fit()

    sgd = StochasticGD(X, y, gradient_OLS, 0.044, 1001)
    theta2 = sgd.fit()

    print(theta)
    print(theta2)

