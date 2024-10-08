import numpy as np
#from typing import Callable

np.random.seed(3)

class GradientDescent:
    """"""
    def __init__(self, X: np.ndarray, Y: np.ndarray, cost=0):
        self.X = X
        self.Y = Y
        self.cost = cost # IF cost function, we should use AutoGrad (not implemented)

    def fit(self, 
            derivative, 
            eta: float, 
            max_iter=1_000,
            tol=1e-6,
            momentum=0):
        
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
        return self.eta*self.gradient + self.change*self.momentum


class StochasticGD(GradientDescent):
    """"""
    def fit(self, 
            derivative, 
            eta: float, 
            epochs=1_000,
            batch_size=16,
            tol=1e-6,
            momentum=0):
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

# Testing

def gradient_OLS(X, y, theta):
    n = X.shape[0]
    return (2.0/n)*X.T @ (X @ theta - y)

# def gradient_Ridge(X, y, beta):
#     llambda = 0.001
#     n = X.shape[0]
#     return 2.0/n*X.T @ (X@beta - y)+2*llambda*beta


if __name__ == "__main__":
    n = 100
    x = 2*np.linspace(-1,1,n)
    y = 1 + 2*x + 3*x**2 #+ np.random.randn(n)

    p = 2
    X = np.c_[*(x**i for i in range(p+1))]
    
    gd = GradientDescent(X, y)
    theta = gd.fit(gradient_OLS, 0.1, 1000)

    sgd = StochasticGD(X, y)
    theta2 = sgd.fit(gradient_OLS, 0.1, 1000, 16)

    print(theta)
    print(theta2)
