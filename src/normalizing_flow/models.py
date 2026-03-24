import torch
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, mask):
        super().__init__()
        self.mask = mask
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, dim * 2) 
        )
        
        # Identity Initialization
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z):
        z_A = z * self.mask
        s, t = self.mlp(z_A).chunk(2, dim=-1)
        
        # Safety clamp
        s = torch.clamp(s, min=-5.0, max=5.0) 
        
        inv_mask = 1.0 - self.mask
        s = s * inv_mask
        t = t * inv_mask
        
        x = z_A + inv_mask * (z * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return x, log_det

class NormalizingFlow(nn.Module):
    def __init__(self, dim=2, num_layers=8, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[::2] = 1.0
            else:
                mask[1::2] = 1.0
            self.layers.append(AffineCouplingLayer(dim, hidden_dim, mask))

    def forward(self, z):
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            layer.mask = layer.mask.to(z.device) 
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total
