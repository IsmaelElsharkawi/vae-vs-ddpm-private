import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block: two 3x3 convs with BatchNorm + ReLU and a skip."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, padding=1),     # 32 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),   # 16 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),   # 8 -> 4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResBlock(512),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = torch.flatten(h, 1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.net = nn.Sequential(
            ResBlock(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),    # 16 -> 32
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 512, 4, 4)
        return self.net(h)


class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)
