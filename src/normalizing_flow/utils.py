import torch
import math
import matplotlib.pyplot as plt

def integrate_and_plot_breit_wigner(flow, target_pdf, num_samples=50000000):
    print("\nStarting Physics Integration...")
    dim = 2
    flow.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, dim)
        log_p_z = -0.5 * (z**2).sum(dim=-1) - (dim / 2.0) * math.log(2 * math.pi)
        
        y, log_det = flow(z)
        log_q_y = log_p_z - log_det
        log_f_y = target_pdf(y)
        
        weights = torch.exp(log_f_y - log_q_y)
        
        integral = weights.mean().item()
        variance = weights.var().item()
        error = math.sqrt(variance / num_samples)
        
        print(f"Calculated Integral : {integral:.6f} +/- {error:.6f}")
        print(f"Variance (sigma^2)  : {variance:.6f}")
        
        sig_y0 = torch.sigmoid(y[:, 0])
        sig_y1 = torch.sigmoid(y[:, 1])
        E = 50.0 + 100.0 * sig_y0
        cos_theta = -1.0 + 2.0 * sig_y1
        
        E_np = E.numpy()
        cos_np = cos_theta.numpy()
    
    m_Z = 91.1876
    plt.figure(figsize=(9, 6))
    plt.hist2d(E_np, cos_np, bins=120, cmap='magma', range=[[80, 105], [-1, 1]])
    plt.colorbar(label='Generated Event Density')
    plt.axvline(m_Z, color='white', linestyle='--', alpha=0.7, label=f'Z Mass ({m_Z} GeV)')
    plt.title("Learned Z Boson Resonance ($e^+ e^- \\to Z \\to \\mu^+ \\mu^-$)")
    plt.xlabel("Invariant Mass $E$ [GeV]")
    plt.ylabel("Scattering Angle $\\cos(\\theta)$")
    plt.legend()
    plt.savefig("breit_wigner_plot.png")
    plt.close()

def integrate_and_plot_rosenbrock(flow, target_pdf, num_samples=5000000):
    print("\nStarting Physics Integration...")
    dim = 8
    flow.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, dim)
        log_p_z = -0.5 * (z**2).sum(dim=-1) - (dim / 2.0) * math.log(2 * math.pi)
        
        y, log_det = flow(z)
        log_q_y = log_p_z - log_det
        log_f_y = target_pdf(y)
        
        weights = torch.exp(log_f_y - log_q_y)
        
        integral = weights.mean().item()
        variance = weights.var().item()
        error = math.sqrt(variance / num_samples)
        
        print(f"Calculated Integral : {integral:.6f} +/- {error:.6f}")
        print(f"Variance (sigma^2)  : {variance:.6f}")
        
        sig_y0 = torch.sigmoid(y[:, 0])
        sig_y1 = torch.sigmoid(y[:, 1])
        x0 = -5.0 + 10.0 * sig_y0
        x1 = -5.0 + 10.0 * sig_y1
        
        x0_np = x0.numpy()
        x1_np = x1.numpy()
    
    plt.figure(figsize=(9, 6))
    plt.hist2d(x0_np, x1_np, bins=150, cmap='magma', range=[[-3, 3], [-1, 5]])
    plt.colorbar(label='Generated Density')
    plt.title("Learned 8D Rosenbrock Distribution (Dimensions 0 and 1)")
    plt.xlabel("$X_0$")
    plt.ylabel("$X_1$")
    plt.savefig("rosenbrock_plot.png")
    plt.close()
