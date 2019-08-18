import numpy as np

class SOM(object):
    def __init__(self, map_sz=10):
        self.map_size = 10
        self.W = None

    def train(self, X, init_lr=0.1, init_sigma=None, max_iter=1000, 
              verbose=True):
        """
        Trains self-organising map
        """
        N = X.shape[0]
        D = X.shape[1]
        print(X.shape)
        S = self.map_size
        self.W = np.random.randn(S, S, D)

        lr = init_lr
        if init_sigma==None:
            init_sigma = S
        sigma = init_sigma

        neigh_func_t = np.arange(1, S+1)
        for it in range(max_iter):
            for i in range(N):
                train_pt = X[i, :]
                diff = np.sum((self.W - train_pt)**2, axis=-1)
                min_idx_i, min_idx_j = np.unravel_index(np.argmin(diff), diff.shape)

                neigh_func = ((neigh_func_t - (min_idx_j+1))**2).reshape(1, S) + ((neigh_func_t - (min_idx_i+1))**2).reshape(S, 1)
                neigh_func = np.exp(-neigh_func / (2*sigma*sigma))

                self.W += lr * neigh_func[:,:,None] * (train_pt - self.W)

            lr = init_lr * np.exp(-it / N)
            sigma = init_sigma * np.exp(-it / (N / np.log(init_sigma)))

            if verbose:
                if it%2==0:
                    print('Iteration {}'.format(it))

        np.save('som_W', self.W)
