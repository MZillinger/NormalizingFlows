import torch
import torch.nn.functional as F
import math

def breit_wigner_pdf(y):
    m_Z = 91.1876     
    Gamma_Z = 2.4952  
    
    sig_y0 = torch.sigmoid(y[:, 0])
    sig_y1 = torch.sigmoid(y[:, 1])
    
    E = 50.0 + 100.0 * sig_y0
    cos_theta = -1.0 + 2.0 * sig_y1
    
    log_jac_0 = math.log(100.0) + F.logsigmoid(y[:, 0]) + F.logsigmoid(-y[:, 0])
    log_jac_1 = math.log(2.0) + F.logsigmoid(y[:, 1]) + F.logsigmoid(-y[:, 1])
    
    log_bounding_jac = log_jac_0 + log_jac_1
    
    s = E**2
    bw_numerator = s
    bw_denominator = (s - m_Z**2)**2 + (m_Z * Gamma_Z)**2
    breit_wigner = bw_numerator / bw_denominator
    
    angular = 1.0 + cos_theta**2
    f_phys = breit_wigner * angular
    
    log_f_y = torch.log(f_phys) + log_bounding_jac
    
    return log_f_y

def rosenbrock_pdf(y):
    # Shape of y: [batch_size, 8]
    x = -5.0 + 10.0 * torch.sigmoid(y)
    
    log_jac = math.log(10.0) + F.logsigmoid(y) + F.logsigmoid(-y)
    log_bounding_jac = log_jac.sum(dim=-1)
    
    x_i = x[:, :-1]   # First 7 dimensions
    x_next = x[:, 1:] # Shifted next 7 dimensions
    
    term1 = 100.0 * (x_next - x_i**2)**2
    term2 = (1.0 - x_i)**2
    
    log_f_phys = -0.05 * (term1 + term2).sum(dim=-1)
    log_f_y = log_f_phys + log_bounding_jac
    
    return log_f_y
