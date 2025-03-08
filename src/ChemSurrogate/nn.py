import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, noise=0.0):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, latent_dim),
            nn.RMSNorm(latent_dim),
            nn.GELU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
        )
        
        self.noise = noise

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        noise = torch.randn_like(z) * self.noise
        z += noise
        x_reconstructed = self.decode(z)
        return x_reconstructed


class Emulator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer, dropout=0.0):
        super(Emulator, self).__init__()
        self.layers = nn.Sequential(
            #Input Layer
            nn.Linear(input_dim, hidden_layer),
            nn.RMSNorm(hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout),

            #Hidden Layer #1
            nn.Linear(hidden_layer, hidden_layer),
            nn.RMSNorm(hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout),

            #Hidden Layer #2
            nn.Linear(hidden_layer, hidden_layer),
            nn.RMSNorm(hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout),

            #Output Layer
            nn.Linear(hidden_layer, output_dim),
            nn.Sigmoid() # Data is normalized between 0 & 1. 
        )
    
    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return x[:, 5:] + self.net(x)


class RecursiveResNet(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=64, num_blocks=4, dropout=0.0):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x