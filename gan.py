import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class GAN(object):
    def __init__(self, noise_dim=100):
        self.noise_dim = noise_dim

        self.discriminator = discriminator()
        self.generator = generator(noise_dim=noise_dim)

    def train(self, X, learn_rate=0.001, n_epochs=5, batch_sz=100,
                verbose=False):
        """
        Performs parameter updates to generator and discriminator networks

        Input:
        - X: Training images of shape (N, 1, 28, 28)

        Output:
        - Dictionary containing training history
        """
        N = X.shape[0]

        dis_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                         lr=learn_rate, betas=(0.5, 0.999))
        gen_optimizer = torch.optim.Adam(self.generator.parameters(),
                                         lr=learn_rate, betas=(0.5, 0.999))

        train_stats = {'dis_loss': [], 'gen_loss': [], 'ite': []}
        iteration = 0
        for epoch in range(n_epochs):
            for bch in range(0, N, batch_sz):
                X_ = X[bch:bch+batch_sz, :]

                # Update parameters of discriminator network
                noise = (2 * torch.rand(batch_sz, self.noise_dim)) - 1

                scores_real = self.discriminator(X_)
                scores_fake = self.discriminator(self.generator(noise))
                # Discriminator loss and parameter updates
                dis_loss = self.discriminator.loss(scores_real, scores_fake)

                dis_optimizer.zero_grad()
                dis_loss.backward()
                dis_optimizer.step()

                # Update parameters of generator network
                noise = (2 * torch.rand(batch_sz, self.noise_dim)) - 1

                scores_fake = self.discriminator(self.generator(noise))
                # Generator loss and parameter updates
                gen_loss = self.generator.loss(scores_fake)

                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()

                if iteration%600==0:
                    # Show images after every epoch
                    self.display_images(self.generator(noise))

                if iteration%100==0:
                    train_stats['dis_loss'].append(dis_loss)
                    train_stats['gen_loss'].append(gen_loss)
                    train_stats['ite'].append(iteration)
                    if verbose:
                        print('Iteration {}'.format(iteration))
                        print('Discriminator loss: {}'.format(dis_loss))
                        print('Generator loss: {}'.format(gen_loss))
                iteration += 1

        return train_stats

    def test(self, N=16):
        """
        Generate images from noise input at test time
        """
        noise = (2 * torch.rand(N, self.noise_dim)) - 1

        return self.generator(noise)

    def display_images(self, img):
        """
        Displays generated images in a grid
        """
        img = img.detach().numpy()
        img = img[0:16, :].reshape(16, 28, 28)

        plt.figure()
        gs = gridspec.GridSpec(4, 4)
        for i, im in enumerate(img):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(im, cmap='gray_r')
        plt.show()


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, X):
        # Input X has dimensions (N, C, H, W)
        x = self.fc1(X.view(X.shape[0], -1))
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)

        return x

    def loss(self, scores_real, scores_fake):
        """
        Binary cross-entropy loss is used here
        """
        neg_abs = -scores_real.abs()
        loss = (scores_real.clamp(min=0) - scores_real * torch.ones(scores_real.shape) + (1 + neg_abs.exp()).log()).mean()

        neg_abs = -scores_fake.abs()
        loss += (scores_fake.clamp(min=0) - scores_fake * torch.zeros(scores_fake.shape) + (1 + neg_abs.exp()).log()).mean()
        
        return loss


class generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(generator, self).__init__()
        self.noise_dim = noise_dim

        self.fc1 = nn.Linear(noise_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 28*28)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, noise):
        # Input X is noise of shape (N, noise_dim)
        x = self.fc1(noise)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)

        return x.view(noise.shape[0], 1, 28, 28)

    def loss(self, scores_fake):
        """
        Binary cross-entropy loss is used here
        """
        neg_abs = -scores_fake.abs()
        loss = scores_fake.clamp(min=0) - scores_fake * torch.ones(scores_fake.shape) + (1 + neg_abs.exp()).log()

        return loss.mean()
