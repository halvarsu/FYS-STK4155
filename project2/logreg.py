import numpy as np

class Logistic_Regression:
    def __init__(self, learn_rate=0.01, iterations=1000):
        self.learn_rate = learn_rate
        self.iterations = iterations
    
    def fit(self, X, y):  
        self.beta = np.zeros(X.shape[1])
        # Update parameters with radient descent
        for i in range(self.iterations):
            z = np.dot(X, self.beta)
            p = self.sigmoid(z)
            grad = (X.T @ (p - y)) #/ y.shape[0]
            self.beta -= self.learn_rate * grad
    
    def predict(self, X, binary=True):
        self.ypred = self.sigmoid(np.dot(X, self.beta))
        if binary:
            return 1*(self.ypred >= 0.5) #Get binary values if over threshold
        else:
            return self.ypred
    
    def accuracy(self, y):
        return np.sum(np.isclose(self.ypred,y))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, p, y):
        return (-y * np.log(p) - (1 - y) * np.log(1 - p))