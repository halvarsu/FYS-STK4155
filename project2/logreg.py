import numpy as np

class Logistic_Regression:
    """
    Logistic regression:
    - Normalize => probabilities
    - lmbd if regularization (L2)
    """
    def __init__(self, learn_rate=0.01, iterations=1000, normalize=False, lmbd=0, tol=1e-5):
        self.learn_rate = learn_rate
        self.iterations = iterations
        self.normalize = normalize
        self.lmbd = lmbd
        self.tolerance=tol
    
    def fit(self, X, y): 
        n = X.shape[1]
        self.beta = np.zeros(n)
        error = []
        # Update parameters with gradient descent
        for i in range(self.iterations):
            z = np.dot(X, self.beta)
            p = self.sigmoid(z)
            p2 = 1*(p>0.5)
            err = np.mean(p2)
            error.append(err)
            if self.normalize:
                grad = (X.T @ (p - y) - self.lmbd*self.beta)
            else:
                grad = (X.T @ (p - y) - self.lmbd*self.beta) 
            self.beta -= self.learn_rate * grad
            #if error < self.tolerance:
            #    print('Converge at i=',i)
            #    break
        return error
    
    def fit2(self, X, y): 
        n = X.shape[1]
        #self.beta = np.zeros(n)
        self.beta = 1e-6*(np.random.random(size = (n,)) - 0.5).squeeze()
        value = np.zeros((self.iterations))
        # Update parameters with gradient descent
        for i in range(self.iterations):
            a = (X @ self.beta).T
            print(np.mean(a))
            p0  =  1/ (1 + np.exp(a)).squeeze()
            p1 = (np.exp(a)*p0).squeeze()
            p = np.choose(y,[p0, p1])

            deriv_cost = X.T @ (y - p)
            self.beta -= self.learn_rate * deriv_cost 

            correct = p > 0.5
            value[i] = np.mean(correct)
        return value
          
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
        return np.sum(-y * np.log(p) - (1 - y) * np.log(1 - p)) #+ (1/self.beta.shape[0])*(0.5*self.lmbd*np.sum(self.beta**2))