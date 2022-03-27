#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: https://github.com/shfoo
"""

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class GAN(object):
    def __init__(self, noise_dim=100, with_labels=False, device=torch.device('cpu')):
        self.noise_dim = noise_dim
        self.with_labels = with_labels
        self.device = device

        self.discriminator = Discriminator(with_labels=with_labels).to(device)
        self.generator = Generator(noise_dim=noise_dim, with_labels=with_labels).to(device)
        
    def fit(self, X, y=None, learning_rate=0.001, num_epochs=10, batch_size=100,
            verbose=False):
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                   lr=learning_rate, betas=(0.5, 0.999))
        generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                               lr=learning_rate, betas=(0.5, 0.999))
        loss_function = nn.BCEWithLogitsLoss()
        
        iteration = 0
        training_stats = {'ite': [], 'discriminator_loss': [], 'generator_loss': []}
        
        for epoch in range(1, num_epochs+1):
            idx = torch.randperm(X.size()[0])
            X, y = X[idx], y[idx] if self.with_labels else None
            
            self.generator.train()
            self.discriminator.train()
            for batch_loc in range(0, X.size()[0], batch_size):
                X_batch = X[batch_loc: batch_loc+batch_size, :]
                y_batch = y[batch_loc: batch_loc+batch_size] if self.with_labels else None
                
                # Update parameters of discriminator network
                noise = ((2 * torch.rand(batch_size, self.noise_dim)) - 1).to(self.device)
                
                scores_real = self.discriminator(X_batch, y_batch if self.with_labels else None)
                scores_fake = self.discriminator(self.generator(noise, y_batch if self.with_labels else None), 
                                                 y_batch if self.with_labels else None)
                # Discriminator loss and parameter updates
                discriminator_loss = loss_function(scores_real, torch.ones(scores_real.shape).to(self.device)) \
                                     + loss_function(scores_fake, torch.zeros(scores_fake.shape).to(self.device))

                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()
                
                # Update parameters of generator network
                noise = ((2 * torch.rand(batch_size, self.noise_dim)) - 1).to(self.device)

                scores_fake = self.discriminator(self.generator(noise, y_batch if self.with_labels else None), 
                                                 y_batch if self.with_labels else None)
                # Generator loss and parameter updates
                generator_loss = loss_function(scores_fake, torch.ones(scores_fake.shape).to(self.device))

                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()
                
                iteration += 1
                
                if iteration % max((X.size()[0] // batch_size) // 5, 1) == 0:
                    training_stats['discriminator_loss'].append(discriminator_loss.item())
                    training_stats['generator_loss'].append(generator_loss.item())
                    training_stats['ite'].append(iteration)
                
                # Show images after every epoch
                if iteration % (X.size()[0] // batch_size) == 0:
                    print('Generated images after epoch {}'.format(epoch))
                    if self.with_labels:
                        labels = torch.zeros((10, 10)).to(self.device)
                        labels[range(10), range(10)] = 1
                    self.display_images(self.generator(noise[0:10, :], 
                                        labels if self.with_labels else None))
                    
            if verbose:
                print('Iteration {}'.format(iteration))
                print('Discriminator loss: {}'.format(discriminator_loss))
                print('Generator loss: {}'.format(generator_loss))
                
        return training_stats
    
    def predict(self, N, y=None):
        noise = ((2 * torch.rand(N, self.noise_dim)) - 1).to(self.device)

        return self.generator(noise, y if self.with_labels else None)
    
    def display_images(self, img):
        """
        Displays generated images in a grid
        """
        try:
            img = img.detach().numpy()
        except:
            img = img.cpu().detach().numpy()
        img = img[0:10, :].reshape(10, 28, 28)

        plt.figure()
        gs = gridspec.GridSpec(2, 5)
        for i, im in enumerate(img):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(im, cmap='gray_r')
        plt.show()
        
        
class Discriminator(nn.Module):
    def __init__(self, with_labels=False):
        super(Discriminator, self).__init__()
        
        self.with_labels = with_labels

        self.fc1 = nn.Linear(28*28 + (10 if with_labels else 0), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, X, y=None):
        # Input X has dimensions (N, C, H, W)
        if self.with_labels:
            x = self.fc1(torch.cat((X.view(X.shape[0], -1), y), dim=1))
        else:
            x = self.fc1(X.view(X.shape[0], -1))    
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)

        return x
    
    
class Generator(nn.Module):
    def __init__(self, noise_dim=100, with_labels=False):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.with_labels = with_labels

        self.fc1 = nn.Linear(noise_dim + (10 if with_labels else 0), 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 28*28)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, noise, y=None):
        # Input X is noise of shape (N, noise_dim)
        if self.with_labels:
            x = self.fc1(torch.cat((noise, y), dim=1))
        else:
            x = self.fc1(noise)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)

        return x.view(noise.shape[0], 1, 28, 28)
