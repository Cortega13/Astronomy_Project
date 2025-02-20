import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, noise=0.0):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input Layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            # Encoder Hidden Layer
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Decoder Hidden Layer
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            #Output Layer
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(), # Data is normalized between 0 & 1. 
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
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout),

            #Hidden Layer #1
            nn.Linear(hidden_layer, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout),

            #Hidden Layer #2
            nn.Linear(hidden_layer, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout),

            #Output Layer
            nn.Linear(hidden_layer, output_dim),
            nn.Sigmoid() # Data is normalized between 0 & 1. 
        )
    
    def forward(self, x):
        return self.layers(x)