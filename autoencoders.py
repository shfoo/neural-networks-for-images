#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: https://github.com/shfoo
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5,
                                               kernel_size=5, stride=1, padding=2), 
                                     nn.LeakyReLU(), 
                                     nn.Conv2d(in_channels=5, out_channels=15,
                                               kernel_size=5, stride=1, padding=2), 
                                     nn.LeakyReLU(), 
                                     nn.Flatten(), 
                                     nn.Linear(15*28*28, 2*latent_dim))
        
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 15*28*28), 
                                     nn.Unflatten(1, (15, 28, 28)), 
                                     nn.ConvTranspose2d(in_channels=15, out_channels=5, 
                                                        kernel_size=5, stride=1, padding=2), 
                                     nn.LeakyReLU(), 
                                     nn.ConvTranspose2d(in_channels=5, out_channels=1, 
                                                        kernel_size=5, stride=1, padding=2), 
                                     nn.LeakyReLU(), 
                                     nn.ConvTranspose2d(in_channels=1, out_channels=1, 
                                                        kernel_size=1, stride=1, padding=0), 
                                     nn.Sigmoid())
        
    def encode(self, X):
        latent_vec = self.encoder(X)
        return latent_vec[:, 0:2], latent_vec[:, 2:4]
    
    def decode(self, z):
        return self.decoder(z)

    def reparameterise(self, means, log_variances):
        return means + torch.exp(log_variances) * torch.normal(mean=0, std=torch.ones(means.shape)).to(means.get_device())

    def fit(self, X, y=None, learning_rate=0.001, num_epochs=10, batch_size=100, 
            kl_loss_weight=0.001,
            verbose=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        reconstruction_loss_function = nn.MSELoss()

        iteration = 0
        training_stats = {'ite': [], 'loss': [], 
                          'reconstruction_loss': [], 'kl_loss': []}
        
        for epoch in range(1, num_epochs+1):
            idx = torch.randperm(X.size()[0])
            X, y = X[idx], y
            
            self.train()
            for batch_loc in range(0, X.size()[0], batch_size):
                X_batch = X[batch_loc: batch_loc+batch_size, :]
                y_batch = None

                means, log_variances = self.encode(X_batch)
                reconstructed_inputs = self.decode(self.reparameterise(means, log_variances))

                reconstruction_loss = reconstruction_loss_function(reconstructed_inputs, X_batch)
                kl_loss = (-0.5 * (-torch.exp(log_variances) - means**2 + log_variances + 1).sum(axis=1)).mean()
                loss = reconstruction_loss + kl_loss_weight * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                iteration += 1

                if iteration % max((X.size()[0] // batch_size) // 5, 1) == 0:
                    self.eval()
                    
                    training_stats['ite'].append(iteration)
                    training_stats['loss'].append(loss.item())
                    training_stats['reconstruction_loss'].append(reconstruction_loss.item())
                    training_stats['kl_loss'].append(kl_loss.item())
                    
                    self.train()
                    
            if verbose:
                print('Loss after epoch {}: {}'.format(epoch, loss))
                
        return training_stats