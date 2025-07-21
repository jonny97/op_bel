import torch
import torch.nn as nn
class StochasticFeatures(nn.Module):
    def __init__(self, kernel, x_dim, num_features, prior = torch.nn.init.normal_):
        super().__init__()
        self.x_dim = x_dim
        self.num_features = num_features
        self.kernel = kernel
        self.w = nn.Linear(x_dim, num_features, bias=False)
        prior(self.w.weight)
        self.w.weight.requires_grad = False

    def forward(self, x):
        if self.kernel == "trig":
            return  torch.sin(self.w(x))
        elif self.kernel == "relu":
            return  torch.nn.functional.relu(self.w(x))
        else:
            raise NotImplementedError("Fourier kernel is not implemented yet")


class ProjectionNets(nn.Module):
    def __init__(self, x_dim, num_learned_basis, num_fixed_basis, learning_hidden_dim, activation = torch.nn.ReLU(), basis=None):
        """
        x_dim: dimension of the input (10)
        num_learned_basis: number of learned basis
        learning_hidden_dim: width of hidden layers for the FNN
        activation: activation function
        num_fixed_basis: number of fixed basis
        basis: a (B, input_dim) tensor of Fourier basis vectors. If None, we initialize randomly.
        """
        super().__init__()
        self.x_dim = x_dim
        self.num_learned_basis = num_learned_basis
        self.learning_hidden_dim = learning_hidden_dim
        self.activation = activation
        self.num_fixed_basis = num_fixed_basis
        self.basis = basis

        # --- 1) define the FNN for the first A outputs ---
        layers = [x_dim] + learning_hidden_dim + [num_learned_basis]
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            # add activation after every hidden layer, but not after the last Linear
            if i < len(layers) - 2:
                modules.append(activation)
        self.fnn = nn.Sequential(*modules)

        # --- 2) prepare the fixed basis ---
        if basis is None:
            self.basis = StochasticFeatures(kernel="trig", x_dim=x_dim, num_features=num_fixed_basis)
        else:
            self.basis = basis

    def forward(self, x):
        """
        x: tensor of shape (..., x_dim)
        returns: tensor of shape (..., num_learned_basis + num_fixed_basis)
        """
        learned_out = self.fnn(x)              
        fixed_out = self.basis(x)
        return torch.cat([learned_out, fixed_out], dim=-1)

def periodic(x):
    y = 2 * torch.pi * x
    return torch.cat(
        [torch.cos(y), torch.sin(y), torch.cos(2 * y), torch.sin(2 * y)], -1
    )


class DeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        x_dim = 1
        y_dim = 4

        self.branch_nets = nn.Sequential(
            nn.Linear(128, 128), # coressponds to xi_i^k
            nn.Tanh(),
            nn.Linear(128, 128), # coressponds to xi_i^k
            nn.Tanh(),
            nn.Linear(128, 128), # coressponds to xi_i^k
            nn.Tanh(),
            nn.Linear(128, 128), # coressponds to c_i^k 
        )

        self.trunk_net = nn.Sequential(
            nn.Linear(y_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )
        
    def forward(self, x, u, y):
        """
        Inputs:
          x: Tensor of shape [B, L, 1], where B = batch size, L = number of integration points per sample.
          u: Tensor of shape [B, L] (function values corresponding to x).
        """
        u_flattened = torch.flatten(u, start_dim=-2)
        branch_results = self.branch_nets(u_flattened)
        z = periodic(y)
        trunk_results = self.trunk_net(z)
        results = (branch_results.unsqueeze(1) * trunk_results).sum(dim=-1, keepdim=True)
        return results
    
    
class BelNet(nn.Module):
    def __init__(self, num_learned_basis = 50, num_fixed_basis = 50, basis = None):
        super().__init__()
        x_dim = 1
        y_dim = 4
        self.projection_nets = ProjectionNets(x_dim, num_learned_basis, num_fixed_basis, learning_hidden_dim=[64,64,64,64], activation=torch.nn.ReLU(), basis=basis)
        self.branch_nets = nn.Sequential(
            nn.Linear(num_learned_basis + num_fixed_basis, 100), # coressponds to xi_i^k
            nn.Tanh(),
            nn.Linear(100, 100), # coressponds to xi_i^k
            nn.Tanh(),
            nn.Linear(100, 100), # coressponds to xi_i^k
            nn.Tanh(),
            nn.Linear(100, 128), # coressponds to c_i^k 
        )

        self.trunk_net = nn.Sequential(
            nn.Linear(y_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )
        
    def forward(self, x, u, y):
        """
        Inputs:
          x: Tensor of shape [B, L, 1], where B = batch size, L = number of integration points per sample.
          u: Tensor of shape [B, L] (function values corresponding to x).
        """
        B, L, _ = x.shape  # B: batch size, L: number of points
        psi_outputs = self.projection_nets(x) # shape: [B, L, R]
        psi_transpose = psi_outputs.transpose(1, 2)  # shape: [B, R, L]
        proj_integration = (torch.bmm(psi_transpose, u) / float(L)).squeeze(-1) # shape: [B, R]
        branch_results = self.branch_nets(proj_integration) # shape: [B, N]
        z = periodic(y)
        trunk_results = self.trunk_net(z)
        results = (branch_results.unsqueeze(1) * trunk_results).sum(dim=-1, keepdim=True)
        return results