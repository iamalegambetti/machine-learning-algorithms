import torch

class KMeans:
    def __init__(self, K, EPOCHS = 1_000):
        self.K = K 
        self.centroids = None
        self.labels = None
        self.EPOCHS = EPOCHS
        
    def fit(self, X):
        
        # set seed to remove 
        torch.manual_seed(46) 
        
        m, n = X.shape
        
        # init centroids and labels 
        self.centroids = X[torch.randint(low = 0, high = m, size = (self.K, 1))].squeeze(1)
        self.labels = torch.zeros((m, 1))
        
        for _ in range(self.EPOCHS):
            
            # STEP: cluster assignment 
            for i in range(m): 
                distances = torch.tensor([self.l2(X[i], c) for c in self.centroids])
                target = torch.argmin(distances) 
                self.labels[i] = target
            
            # STEP: update centroid
            for c in self.labels.unique():
    
                indices = torch.where(self.labels == c)[0]
                centroids_c = X[indices]
                new_centroid = torch.mean(centroids_c, axis = 0)

                self.centroids[torch.tensor(c, dtype = torch.long)] = new_centroid
            
    
    @staticmethod
    def l2(x1, x2):
        return torch.sum(torch.pow(torch.subtract(x1, x2), 2))
    
    
    def predict(self, X):
        m, n = X.shape
        labels_predict = torch.zeros((m, 1))
        for i in range(m): 
            distances = torch.tensor([self.l2(X[i], c) for c in self.centroids])
            target = torch.argmin(distances) 
            labels_predict[i] = target
        return labels_predict
            