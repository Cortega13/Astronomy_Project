import torch
import torch.nn as nn
import torchode as tode
import numpy as np
from ChemSurrogate.configs import (
    ModelConfig,
)


class A(nn.Module):
    def __init__(self, input_dim, z_dim, dropout_rate=0.1):
        super(A, self).__init__()

        self.z_dim = z_dim
        self.dropout_rate = dropout_rate

        hidden_dim1 = z_dim
        out_dim = z_dim**2
        hidden_dim2 = out_dim//2

        self.layer_in = nn.Linear(input_dim, hidden_dim1)
        self.layer_hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer_out = nn.Linear(hidden_dim2, out_dim)

        nn.init.kaiming_normal_(self.layer_in.weight, a=0.2)
        nn.init.kaiming_normal_(self.layer_hidden.weight, a=0.2)
        nn.init.kaiming_normal_(self.layer_out.weight, a=0.2)
        
        scale = 0.1
        bias = torch.diag(-torch.ones(z_dim) * scale)
        self.layer_out.bias.data = bias.flatten()
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, p):
        h = self.LeakyReLU(self.layer_in(p))
        h = self.dropout(h)  # Apply dropout after activation
        
        h = self.LeakyReLU(self.layer_hidden(h))
        h = self.dropout(h)  # Apply dropout after activation
        
        h = self.LeakyReLU(self.layer_out(h))
        # No dropout on the final output as it's reconstructing the matrix A
        
        return h.reshape(self.z_dim, self.z_dim)


class B(nn.Module):
    def __init__(self, input_dim, z_dim, dropout_rate=0.1):
        super(B, self).__init__()

        self.z_dim = z_dim
        self.dropout_rate = dropout_rate

        hidden_dim1 = z_dim
        out_dim = z_dim**3
        hidden_dim2 = int(np.sqrt(out_dim))
        hidden_dim3 = out_dim//2

        self.layer_in = nn.Linear(input_dim, hidden_dim1)
        self.layer_hidden1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer_hidden2 = nn.Linear(hidden_dim2, hidden_dim3)
        self.layer_out = nn.Linear(hidden_dim3, out_dim)
        
        nn.init.kaiming_normal_(self.layer_in.weight, a=0.2)
        nn.init.kaiming_normal_(self.layer_hidden1.weight, a=0.2)
        nn.init.kaiming_normal_(self.layer_hidden2.weight, a=0.2)
        nn.init.kaiming_normal_(self.layer_out.weight, a=0.2)
        self.layer_out.weight.data *= 0.1 
            
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, p):
        h = self.LeakyReLU(self.layer_in(p))
        h = self.dropout(h)  # Apply dropout after activation
        
        h = self.LeakyReLU(self.layer_hidden1(h))
        h = self.dropout(h)  # Apply dropout after activation
        
        h = self.LeakyReLU(self.layer_hidden2(h))
        h = self.dropout(h)  # Apply dropout after activation
        
        h = self.LeakyReLU(self.layer_out(h))
        # No dropout on the final output tensor
        
        return h.reshape(self.z_dim, self.z_dim, self.z_dim)


class LatentODEFunction(nn.Module):
    '''
    The G class gives the evolution in latent space.
        
        g(z:t)_i = C_i  + A_ij * z_j(t) + B_ijk * z_j(t) * z_k(t)
        with einstein summation.
        
            Here 
                - z(t) are the encoded species + physical parameters
                - C is a vector with adjustable/trainable elements (1D), constant term
                - A is a matrix with adjustable/trainable elements (2D)
                - B is a tensor with adjustable/trainable elements (3D)
    '''
    def __init__(self, z_dim):
        '''
        Initialising the tensors C, A and B.
        '''
        super(LatentODEFunction, self).__init__()
        self.C = nn.Parameter(torch.zeros(z_dim).requires_grad_(True))
        
        A_init = torch.randn(z_dim, z_dim) * 0.1
        U, _, V = torch.svd(A_init)
        D = torch.diag(torch.rand(z_dim) * -0.5)
        A_init = U @ D @ V.t()
        
        self.A = nn.Parameter(A_init.requires_grad_(True))
        self.B = nn.Parameter((torch.randn(z_dim, z_dim, z_dim) * 0.01).requires_grad_(True))

    def forward(self, t, z): # t is added here so that the ODE solver can use it.
        '''
        Forward function of the G class, einstein summations over indices.
        '''
        return self.C + torch.einsum("ij, bj -> bi", self.A, z) + torch.einsum("ijk, bj, bk -> bi", self.B, z, z)


class LatentODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, jit_solver, dropout=0.0):
        super(LatentODE, self).__init__()
        
        self.encoder = nn.Sequential(
            # Input Layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            
            # Encoder Hidden Layer
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Sigmoid(),
            nn.Dropout(p=dropout),
        )
        self.jit_solver = jit_solver
        
        self.decoder = nn.Sequential(
            # Decoder Hidden Layer
            nn.Linear(latent_dim+4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            #Output Layer
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(), # Data is normalized between 0 & 1. 
        )
    
    def forward(self, t, p, x, is_training=True):
        z = self.encoder(x)
        
        z = torch.cat([z, p], dim=1)
        
        if is_training:
            z_noised = z + torch.randn_like(z) * ModelConfig.noise
        else:
            z_noised = z

        problem = tode.InitialValueProblem(
            y0=z_noised,
            t_eval=t
        )
        
        z = self.jit_solver.solve(problem).ys
        
        z_reshaped = z.reshape(z.size(0) * z.size(1), z.size(2))
        
        x_recon_reshaped = self.decoder(z_reshaped)
        
        x_recon = x_recon_reshaped.view(z.size(0), z.size(1), -1)
        
        return x_recon
