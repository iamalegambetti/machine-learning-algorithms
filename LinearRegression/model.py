import torch

class LinearRegression:
    def __init__(self, epochs = 1000, learning_rate = 0.01):
        self.EPOCHS = epochs
        self.learning_rate = learning_rate
        self.theta = None 
        
    def fit(self, X, y):
        self.theta = torch.zeros(X.shape[1], 1)
        self.m = X.shape[0]
        for _ in range(self.EPOCHS):
            grad = torch.matmul(X.T , torch.matmul(X, self.theta) - y)
            self.theta = self.theta - (1/self.m) * self.learning_rate * grad
    
    def predict(self, X):
        return torch.matmul(X, self.theta)    