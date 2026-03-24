import torch
import torch.nn as nn
import torch.optim as optim
import math

def train_flow(flow, target_pdf, dim=2, epochs=5000, batch_size=8192, lr=2e-3, anneal_epochs=2000, patience_limit=300):
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    
    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    beta_start = 0.05       
    beta_end = 1.0          
    
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting Training for {epochs} epochs with Early Stopping Monitor...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        beta = beta_start + (beta_end - beta_start) * min(1.0, epoch / anneal_epochs)
            
        z = torch.randn(batch_size, dim)
        log_p_z = -0.5 * (z**2).sum(dim=-1) - (dim / 2.0) * math.log(2 * math.pi)
        
        x, log_det = flow(z)
        log_q_x = log_p_z - log_det
        log_f_x = target_pdf(x)
        
        loss = (log_q_x - beta * log_f_x).mean()
        loss.backward()
        
        # Gradient Clipping
        nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        current_loss = loss.item()
        current_lr = scheduler.get_last_lr()[0]
        
        if epoch % 400 == 0:
            print(f"Epoch {epoch:4d} | Beta: {beta:.3f} | LR: {current_lr:.5f} | Loss: {current_loss:.4f}")
            
        if epoch > anneal_epochs:
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience_limit:
                print(f"\n[EARLY STOPPING TRIGGERED]")
                print(f"Loss completely flatlined for {patience_limit} epochs.")
                print(f"Training successfully terminated early at Epoch {epoch} to save time.")
                break

    return flow
