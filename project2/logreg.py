import numpy as np

class Logistic_Regression:
    """
    Logistic regression:
    - Normalize => probabilities
    - lmbd if regularization (L2)
    """
    def __init__(self, learn_rate=0.01, iterations=1000, normalize=False, lmbd=0):
        self.learn_rate = learn_rate
        self.iterations = iterations
        self.normalize = normalize
        self.lmbd = lmbd
    
    def fit(self, X, y):  
        self.beta = np.zeros(X.shape[1])
        # Update parameters with radient descent
        for i in range(self.iterations):
            z = np.dot(X, self.beta)
            p = self.sigmoid(z)
            if self.normalize:
                grad = (X.T @ (p - y) - self.lmbd*self.beta) / y.shape[0]
            else:
                grad = (X.T @ (p - y)) - self.lmbd*self.beta
            self.beta -= self.learn_rate * grad
    
    def predict(self, X, binary=True):
        self.ypred = self.sigmoid(np.dot(X, self.beta))
        if binary:
            return 1*(self.ypred >= 0.5) #Get binary values if over threshold
        else:
            return self.ypred
    
    def accuracy(self, y, y2):
        if y.shape[0] == y2.shape[0]:
            return np.sum(np.isclose(y,y2)) / y.shape[0]
        else: 
            raise ValueError('Arrays must have the same size')
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, p, y):
        return (-y * np.log(p) - (1 - y) * np.log(1 - p) - 0.5*self.lmbd*(self.beta**2))