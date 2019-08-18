import numpy as np
import torch
import torch.nn as nn

class TwoLayerNet(object):
    def __init__(self, input_dim, n_classes, n_hidden=5):
        self.n_hidden = n_hidden

        self.W1 = 0.05 * np.random.randn(input_dim, n_hidden)
        self.b1 = np.zeros(n_hidden)
        self.W2 = 0.05 * np.random.randn(n_hidden, n_classes)
        self.b2 = np.zeros(n_classes)

    def one_pass(self, X, y=None, reg=0):
        """
        Performs one forward and backward pass
        
        Inputs:
        - X: Single batch of training examples, of shape (N, D)
        - y: Single batch of training labels, of shape (N, )
        - reg: Regularization strength of L2 regularization
        - dropout: Dropout parameter

        Outputs:
        - scores (if y is None): Scores for prediction
        - (loss, grads) if y not None: Current loss and gradients
        """
        W1, b1 = self.W1, self.b1
        W2, b2 = self.W2, self.b2
        N, D = X.shape
        C = W2.shape[1]

        # Forward pass through network
        X_hidden = np.maximum(0, np.dot(X, W1) + b1)
        scores = np.dot(X_hidden, W2) + b2

        if y is None:
            return scores

        # Compute class probabilities (softmax layer)
        class_probs = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(N, 1)
        loss = np.sum(-np.log(class_probs)[range(N), y]) / N
        # Adding L2 regularization term
        loss += reg * np.sum(W1*W1) + reg * np.sum(W2*W2)

        grads = {}
        # Backward pass through network
        grad_scores = class_probs
        grad_scores[range(N), y] -= 1
        grad_scores /= N

        grads['W2'] = np.dot(X_hidden.T, grad_scores) + 2*reg*W2
        grads['b2'] = np.sum(grad_scores, axis=0)

        grad_hidden = np.dot(grad_scores, W2.T)
        grad_hidden[X_hidden <= 0] = 0

        grads['W1'] = np.dot(X.T, grad_hidden) + 2*reg*W1
        grads['b1'] = np.sum(grad_hidden, axis=0)

        return loss, grads

    def train(self, X, y, learn_rate=0.1, n_epochs=10, batch_sz=100,
                verbose=False):
        """
        Inputs:
        - X: Training set of shape (N, D)
        - y: Training labels of shape (N, )
        """
        N = X.shape[0]

        train_stats = {'loss': [], 'train_acc': [], 'ite': []}

        for epoch in range(n_epochs):
            for bch in range(0, N, batch_sz):
                X_ = X[bch:bch+batch_sz, :]
                y_ = y[bch:bch+batch_sz] 

                loss, grads = self.one_pass(X=X_, y=y_)

                # Update network parameters
                self.W1 -= learn_rate*grads['W1']
                self.b1 -= learn_rate*grads['b1']
                self.W2 -= learn_rate*grads['W2']
                self.b2 -= learn_rate*grads['b2']

            y_pred = self.test(X=X)
            train_acc = np.count_nonzero(y_pred==y) / N
            train_stats['loss'].append(loss)
            train_stats['train_acc'].append(train_acc)
            train_stats['ite'].append(epoch)
            if verbose:
                print('Loss: {}, Training accuracy: {}'.format(loss, train_acc))

        return train_stats

    def test(self, X):
        """
        Predicts class labels

        Input:
        - X: Test images of shape (N, D)

        Output:
        - y_pred: Class predictions, of shape (N, )
        """
        scores = self.one_pass(X)
        class_probs = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(X.shape[0], 1)
        y_pred = np.argmax(class_probs, axis=1)

        return y_pred


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input shape (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5,
                                kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Shape: (N, 5, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=15,
                                kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Shape: (N, 15, 7, 7)
        self.fc1 = nn.Linear(15*7*7, 100)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, X):
        x = self.relu1(self.conv1(X))
        x = self.maxpool1(x)

        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)
        scores = self.fc2(x)

        return scores

    def train(self, X, y, learn_rate=0.001, n_epochs=10, batch_sz=100,
                verbose=True):
        """
        Inputs:
        - X: Training set of shape (N, 1, 28, 28)
        - y: Training labels of shape (N, )

        Output:
        - Dictionary containing training history
        """
        N = X.size()[0]
        X = torch.autograd.Variable(X)
        y = torch.autograd.Variable(y)

        optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)
        loss_f = nn.CrossEntropyLoss()

        train_stats = {'loss': [], 'ite': []}

        for epoch in range(n_epochs):
            for bch in range(0, N, batch_sz):
                X_ = X[bch:bch+batch_sz, :]
                y_ = y[bch:bch+batch_sz]

                scores = self.forward(X_)
                loss = loss_f(scores, y_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_stats['loss'].append(loss)
            train_stats['ite'].append(epoch)
            if verbose:
                print('Loss: {}'.format(loss))

        return train_stats


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
                      input_size = 28,
                      hidden_size = 64,
                      num_layers = 1,
                      batch_first = True)

        self.fc = nn.Linear(64, 10)

    def forward(self, X):
        # Input shape (N, T, D)
        out, (h_n, h_c) = self.rnn(X, None)
        # Use last time step for classification
        scores = self.fc(out[:, -1, :])

        return scores

    def train(self, X, y, learn_rate=0.01, n_epochs=1, batch_sz=100,
                verbose=True):
        N = X.size()[0]
        X = torch.autograd.Variable(X)
        y = torch.autograd.Variable(y)

        optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)
        loss_f = nn.CrossEntropyLoss()

        train_stats = {'loss': [], 'ite': []}

        iteration = 0
        for epoch in range(n_epochs):
            for bch in range(0, N, batch_sz):
                X_ = X[bch:bch+batch_sz, :, :]
                y_ = y[bch:bch+batch_sz]

                scores = self.forward(X_)
                loss = loss_f(scores, y_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iteration%100 == 0:
                    train_stats['loss'].append(loss)
                    train_stats['ite'].append(iteration)
                    if verbose:
                        print('Loss: {}'.format(loss))
                iteration += 1

        return train_stats
