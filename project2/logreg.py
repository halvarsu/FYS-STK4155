import numpy as np

class Logistic_Regression:
    """
    Logistic regression:
    - lmbd if regularization (L2)
    """
    def __init__(self, learn_rate=0.01, iterations=1000, lmbd=0, tol=1e-5):
        self.learn_rate = learn_rate
        self.iterations = iterations
        self.lmbd = lmbd
        self.tolerance=tol
    
    def fit(self, X, y):
        """
        Fit the data {X,y} using gradient descent.
        The parameters are first created randomly.
        """
        n = X.shape[1]
        self.beta = 0.01*np.random.random(size=n)
        error = []
        c2 = 0            
        # Update parameters with gradient descent
        for i in range(self.iterations):
            z = X@self.beta
            p = self.sigmoid(z)
            c1 = self.cost(p,y)
            conv = np.abs(c1-c2)
            error.append(c1/y.shape[0])
            if conv < self.tolerance:
                print('Conversion at i=',i)
                break
            grad = (X.T @ (p - y) - 2*self.lmbd*self.beta)/n
            self.beta = self.beta- self.learn_rate * grad
            c2 = c1 
        return error
          
    def predict(self, X):
        """
        Predict y-values for a given set X
        """
        self.ypred = self.sigmoid(X@self.beta)
        return 1*(self.ypred >= 0.5) #Get binary values if over threshold

    
    def accuracy(self, y, y2):
        """
        Calculate the accuracy score of a prediction
        """
        if y.shape[0] == y2.shape[0]:
            return np.sum(np.equal(y,y2)) / y.shape[0]
        else: 
            raise ValueError('Arrays must have the same size')
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    

    def cost(self, p, y):
        """
        Log-likelihood, cost function.
        """
        return -y.T @ np.log(p) - (1 - y.T) @ np.log(1 - p) + np.sum(self.lmbd*np.sum(self.beta**2))