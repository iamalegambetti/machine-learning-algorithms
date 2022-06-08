import torch

class GNN:
    
    def __init__(self, learning_rate = 0.01, EPOCHS = 1000):
        self.learning_rate = learning_rate
        self.EPOCHS = EPOCHS
        self.W1 = None
        self.W2 = None
    
    def fit(self, X, A, y):
        
        """
        One hidden layer GNN with ... hidden neurons.
        Model intended for a binary classifier. 
        
        Input: 
            - X: features matrix of shape (m, n)
            - A: adjacent matrix of shape (m, m)
            - y: binary labels of shape (m, 1)
        """
        # get shapes
        m, n = X.shape
        
        # init weights randomly 
        self.W1, self.W2 = torch.randn(n, n) * 0.01, torch.randn(n, 1) * 0.01
        
        # add self-loops to the adjacent matrix 
        A = torch.add(A, torch.eye(m))
        
        # optimize 
        for _ in range(self.EPOCHS):
            
            # forward pass
            H1 = torch.relu(torch.matmul(A, torch.matmul(X, self.W1))) # hidden layer 
            H2 = torch.matmul(A, torch.matmul(H1, self.W2)) # out layer 
            Y_hat = torch.sigmoid(H2) # activated out layer 
                       
            # backpropagation
            dW2 = torch.matmul(H1.T, torch.subtract(Y_hat, y)) * (1/m)
            dW1 = torch.matmul((torch.matmul(torch.subtract(Y_hat, y), self.W2.T) * (1 - H1 ** 2)).T, X) * (1/m)
         
            # gradient update 
            self.W2 = self.W2 - (1/m) * self.learning_rate * dW2
            self.W1 = self.W1 - (1/m) * self.learning_rate * dW1
    
    def predict(self, X, A, thres = 0.5):
        H1 = torch.relu(torch.matmul(A, torch.matmul(X, self.W1))) 
        H2 = torch.matmul(A, torch.matmul(H1, self.W2)) 
        return torch.where(torch.sigmoid(H2) > 0.5, 1, 0)