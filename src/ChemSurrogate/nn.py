import torch
import torch.nn as nn
import torchode as tode
import numpy as np


class A(nn.Module):
    """
    Neural network that constructs a matrix A from the output layer, 
    starting from the physical input of the chemistry model.
    """
    def __init__(self, input_dim, z_dim):
        super(A, self).__init__()

        self.z_dim = z_dim

        hidden_dim1 = z_dim
        out_dim = z_dim**2
        hidden_dim2 = out_dim//2

        self.layer_in = nn.Linear( input_dim, hidden_dim1)
        self.layer_hidden = nn.Linear(hidden_dim1,hidden_dim2)
        self.layer_out = nn.Linear(hidden_dim2, out_dim)

        self.layer_out.weight.data = torch.zeros_like(self.layer_out.weight)
        bias = torch.diag(-torch.ones(z_dim))
        self.layer_out.bias.data = bias.ravel()
        self.layer_hidden.weight.requires_grad_(True)
        self.layer_hidden.bias.requires_grad_(True)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, p):
        h = self.LeakyReLU(self.layer_in(p))
        h = self.LeakyReLU(self.layer_hidden(h))
        h = self.LeakyReLU(self.layer_out(h))
        return h.reshape(self.z_dim,self.z_dim)

class B(nn.Module):
    """
    Neural network that constructs a tensor B from the output layer, 
    starting from the physical input of the chemistry model.
    """
    def __init__(self, input_dim, z_dim):
        super(B, self).__init__()

        self.z_dim = z_dim

        hidden_dim1 = z_dim
        out_dim = z_dim**3
        hidden_dim2 = int(np.sqrt(out_dim))
        hidden_dim3 = out_dim//2

        self.layer_in = nn.Linear( input_dim, hidden_dim1)
        self.layer_hidden1 = nn.Linear(hidden_dim1,hidden_dim2)
        self.layer_hidden2 = nn.Linear(hidden_dim2,hidden_dim3)
        self.layer_out = nn.Linear(hidden_dim3, out_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, p):
        h = self.LeakyReLU(self.layer_in(p))
        h = self.LeakyReLU(self.layer_hidden1(h))
        h = self.LeakyReLU(self.layer_hidden2(h))
        h = self.LeakyReLU(self.layer_out(h))
        return h.reshape(self.z_dim,self.z_dim,self.z_dim)
    

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
        self.C = nn.Parameter(torch.randn(z_dim).requires_grad_(True))
        self.A = nn.Parameter(torch.randn(z_dim, z_dim).requires_grad_(True))
        self.B = nn.Parameter(torch.randn(z_dim, z_dim, z_dim).requires_grad_(True))

    def forward(self,t, z):     ## t has also be given to the forward function, in order that the ODE solver can read it properly
        '''
        Forward function of the G class, einstein summations over indices.
        '''
        return self.C + torch.einsum("ij, bj -> bi", self.A, z) + torch.einsum("ijk, bj, bk -> bi", self.B, z, z)
    

class LatentODE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(LatentODE, self).__init__()
        
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
        self.ode_func = LatentODEFunction(latent_dim)
        
        self.decoder = nn.Sequential(
            # Decoder Hidden Layer
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            #Output Layer
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(), # Data is normalized between 0 & 1. 
        )
    
    def forward(self, x, time_steps):
        z0 = self.encoder(x)
        zt = odeint(self.ode_func, z0, time_steps)
        x_recon = self.decoder(zt[-1], time_steps)
        return x_recon
    

self.odeterm = tode.ODETerm(self.model, with_args=False)

self.step_method          = tode.Dopri5(term=self.odeterm)
self.step_size_controller = tode.IntegralController(atol=atol, rtol=rtol, term=self.odeterm)
self.adjoint              = tode.AutoDiffAdjoint(self.step_method, self.step_size_controller).to(self.DEVICE) # type: ignore

self.jit_solver = torch.compile(self.adjoint)



## Create initial value problem
problem = tode.InitialValueProblem(
    y0     = z_0.to(self.DEVICE),  
    t_eval = tstep.view(z_0.shape[0],-1).to(self.DEVICE),
)

solution = self.jit_solver.solve(problem, args=p)