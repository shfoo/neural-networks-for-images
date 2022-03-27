#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: https://github.com/shfoo
"""

import torch
import torch.nn as nn


class BaseClassifier(nn.Module):
    def __init__(self):
        super(BaseClassifier, self).__init__()
        
    def fit(self, X, y, learning_rate=0.001, num_epochs=10, batch_size=100, 
            verbose=True, report_train_acc=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()
        
        iteration = 0
        training_stats = {'ite': [], 'loss': [], 'train_acc': []}
        
        for epoch in range(1, num_epochs+1):
            idx = torch.randperm(X.size()[0])
            X, y = X[idx], y[idx]
            
            self.train()
            for batch_loc in range(0, X.size()[0], batch_size):
                X_batch = X[batch_loc: batch_loc+batch_size, :]
                y_batch = y[batch_loc: batch_loc+batch_size]

                scores = self.forward(X_batch)
                loss = loss_function(scores, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                iteration += 1

                if iteration % max((X.size()[0] // batch_size) // 5, 1) == 0:
                    self.eval()
                    
                    training_stats['ite'].append(iteration)
                    training_stats['loss'].append(loss.item())
                    
                    if report_train_acc:
                        training_stats['train_acc'].append(((self.predict(X) == y).sum() / len(y)).item())
                    
                    self.train()
                    
            if verbose:
                print('Loss after epoch {}: {}'.format(epoch, loss))
                
        return training_stats
    
    def predict(self, X, MC_dropout=False, MC_iterations=100):
        if MC_dropout:
            for layer in self.modules():
                if isinstance(layer, nn.Dropout):
                    layer.train()
                else:
                    layer.eval()
                    
            with torch.no_grad():
                class_proba = torch.stack([nn.functional.softmax(self.forward(X), dim=1) for ite in range(MC_iterations)], dim=2)
                
                ypred = torch.argmax(torch.mean(class_proba, dim=2), dim=1)
                pred_variances = torch.var(class_proba, dim=2)[torch.arange(X.size()[0]), ypred]
            
            return ypred, pred_variances
        else:
            self.eval()
            with torch.no_grad():
                ypred = torch.argmax(self.forward(X), dim=-1)
            
            return ypred


class MultiLayerPerceptron(BaseClassifier):
    def __init__(self, input_dim=1*28*28, 
                 num_classes=10, 
                 num_hidden_neurons=[5], 
                 batch_norm=False, dropout_proba=None):
        super(MultiLayerPerceptron, self).__init__()
        
        self.layers = nn.ModuleList([])
        for i in range(0, len(num_hidden_neurons)):
            # Fully-connected layer
            if i==0:
                self.layers.append(nn.Linear(input_dim, num_hidden_neurons[0]))
            else:
                self.layers.append(nn.Linear(num_hidden_neurons[i-1], num_hidden_neurons[i]))
            # Batch normalisation
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(num_hidden_neurons[i]))
            # ReLU non-linearity
            self.layers.append(nn.ReLU())
            # Dropout layer
            if dropout_proba is not None:
                self.layers.append(nn.Dropout(dropout_proba))
        # Final fully-connected layer
        self.layers.append(nn.Linear(num_hidden_neurons[i], num_classes))
                
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        scores = X
            
        return scores
    
    
class CNN(BaseClassifier):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv_layers1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5,
                                                    kernel_size=5, stride=1, padding=2), 
                                          nn.BatchNorm2d(5), 
                                          nn.ReLU(), 
                                          nn.MaxPool2d(kernel_size=2))
        
        self.conv_layers2 = nn.Sequential(nn.Conv2d(in_channels=5, out_channels=15,
                                                    kernel_size=5, stride=1, padding=2),
                                          nn.BatchNorm2d(15), 
                                          nn.ReLU(), 
                                          nn.MaxPool2d(kernel_size=2))
        
        self.fc_layers = nn.Sequential(nn.Linear(15*7*7, 100), 
                                       #nn.BatchNorm1d(100), 
                                       nn.ReLU(), 
                                       nn.Linear(100, 10))

    def forward(self, X):
        x = self.conv_layers1(X)
        x = self.conv_layers2(x)
        scores = self.fc_layers(x.view(x.shape[0], -1))

        return scores
    
    
class LSTM(BaseClassifier):
    def __init__(self):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size = 28,
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
