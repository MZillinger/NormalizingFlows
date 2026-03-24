import sys
import os

# Ensure src module is discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from normalizing_flow import (
    NormalizingFlow,
    breit_wigner_pdf,
    rosenbrock_pdf,
    train_flow,
    integrate_and_plot_breit_wigner,
    integrate_and_plot_rosenbrock
)

def main():
    print("=== Testing Breit-Wigner Flow ===")
    flow_bw = NormalizingFlow(dim=2, num_layers=8, hidden_dim=64)
    # Run a very short training to verify it executes
    trained_bw = train_flow(flow_bw, breit_wigner_pdf, dim=2, epochs=5, batch_size=1024, lr=2e-3, anneal_epochs=2, patience_limit=3)
    
    print("\n=== Testing Rosenbrock Flow ===")
    flow_rb = NormalizingFlow(dim=8, num_layers=16, hidden_dim=128)
    trained_rb = train_flow(flow_rb, rosenbrock_pdf, dim=8, epochs=5, batch_size=1024, lr=2e-3, anneal_epochs=2, patience_limit=3)
    
    print("\nAll tests passed successfully. The models compile and the training loops execute!")

if __name__ == "__main__":
    main()
