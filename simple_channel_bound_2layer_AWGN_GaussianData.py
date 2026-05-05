import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import math


# ==========================================
# 1. Configuration and Environment Setup
# ==========================================
# Use requested device logic
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
SIM_NAME = "simple_channel_bound_2layer_AWGN_GaussianData"
RESULTS_DIR = os.path.join("./results", SIM_NAME)
DATA_DIR = os.environ.get('DATASET', './data')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Hyperparameters
NUM_SAMPLES = 2000
INPUT_DIM = 10       # Reduced to ensure Batch Size > Total Dimensions
CHANNEL_DIM = 4      # Reduced for full-rank covariance matrices
EPOCHS = 150
BATCH_SIZE = 256     # Increased to ensure sufficient samples for empirical covariance
LR = 0.01            # Learning rate for optimization

# Theorem Hyperparameters
SIGMA_TR = 1       # Training AWGN standard deviation
SIGMA_TE = 5       # Inference AWGN standard deviation (Noisier)
PRIOR_SIGMA = 1.0    # Prior standard deviation for weights
K_PARAM = 100.0      # 'k' parameter in the bound
EPSILON = 0.05       # Probability epsilon
SIGMA_0 = 0.5        # Sub-Gaussian parameter for P_{W'|S} x P_Z
SIGMA_LOSS = 0.5     # Sub-Gaussian parameter for P_Z

# ==========================================
# 2. Dataset Generation
# ==========================================
def generate_synthetic_data(num_samples, input_dim):
    """
    Synthesizes linearly separable data with added Gaussian noise.
    """
    torch.manual_seed(42)
    # True weights for data generation
    W_true = torch.randn(input_dim, 1)
    
    # Generate X
    X = torch.randn(num_samples, input_dim)
    
    # Linear combination + Heavy Gaussian noise to prevent 100% accuracy
    noise = torch.randn(num_samples, 1) * 2.5
    y_continuous = torch.matmul(X, W_true) + noise
    
    # Threshold to create binary classification labels
    Y = (y_continuous > 0).float()
    
    # Split into train and test
    split_idx = int(0.8 * num_samples)
    X_train, y_train = X[:split_idx], Y[:split_idx]
    X_test, y_test = X[split_idx:], Y[split_idx:]
    
    # Save dataset to requested path
    data_path = os.path.join(DATA_DIR, f"{SIM_NAME}_dataset.pt")
    torch.save({'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}, data_path)
    print(f"Dataset saved to {data_path}")
    
    return X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)

# ==========================================
# 3. Bayesian Neural Network Modules
# ==========================================
class BayesianLinear(nn.Module):
    """A linear layer with Variational Bayesian inference (Diagonal Gaussian Posterior)"""
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        # Mean and log-variance parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3, -2)) # Init with small variance
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-3, -2))

    def forward(self, x):
        B = x.size(0)
        # Reparameterization trick
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # Sample weights independently FOR EVERY ITEM IN THE BATCH
        epsilon_w = torch.randn(B, self.out_features, self.in_features, device=x.device)
        epsilon_b = torch.randn(B, self.out_features, device=x.device)
        
        weight = self.weight_mu.unsqueeze(0) + weight_sigma.unsqueeze(0) * epsilon_w
        bias = self.bias_mu.unsqueeze(0) + bias_sigma.unsqueeze(0) * epsilon_b
        
        # Batch matrix multiplication: (B, out, in) @ (B, in, 1) -> (B, out, 1)
        out = torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1) + bias
        
        # Flatten the sampled weights for this batch to compute empirical covariance later
        flat_w = weight.view(B, -1)
        flat_b = bias.view(B, -1)
        flat_params = torch.cat([flat_w, flat_b], dim=1)
        
        return out, flat_params

    def kl_divergence(self):
        """Computes exact KL divergence between posterior N(mu, sigma^2) and prior N(0, prior_sigma^2)"""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        kl_weight = 0.5 * torch.sum(
            2 * torch.log(torch.tensor(self.prior_sigma, device=weight_sigma.device)) - 2 * torch.log(weight_sigma) 
            + (weight_sigma**2 + self.weight_mu**2) / (self.prior_sigma**2) - 1
        )
        kl_bias = 0.5 * torch.sum(
            2 * torch.log(torch.tensor(self.prior_sigma, device=bias_sigma.device)) - 2 * torch.log(bias_sigma) 
            + (bias_sigma**2 + self.bias_mu**2) / (self.prior_sigma**2) - 1
        )
        return kl_weight + kl_bias

class WirelessEdgeNet(nn.Module):
    def __init__(self, input_dim, channel_dim, prior_sigma=1.0):
        super().__init__()
        # Layer 1: Transmitter
        self.tx_layer = BayesianLinear(input_dim, channel_dim, prior_sigma)
        # Layer 2: Receiver
        self.rx_layer = BayesianLinear(channel_dim, 1, prior_sigma)
        self.channel_dim = channel_dim

    def forward(self, x, channel_sigma):
        B = x.size(0)
        # Tx processing
        z_tx, tx_params = self.tx_layer(x)
        
        # Wireless AWGN Channel
        # Sample channel noise FOR EVERY ITEM IN THE BATCH
        noise = torch.randn_like(z_tx) * channel_sigma
        z_rx = z_tx + noise
        
        # Rx processing
        out, rx_params = self.rx_layer(z_rx)
        
        # Concatenate all sampled weights across layers
        all_weights = torch.cat([tx_params, rx_params], dim=1)
        
        return out, all_weights, noise

    def get_total_kl(self):
        return self.tx_layer.kl_divergence() + self.rx_layer.kl_divergence()

# ==========================================
# 4. Theorem Components & Loss
# ==========================================
def compute_channel_kl(d, sigma_tr, sigma_te):
    """Exact KL divergence between training channel N(0, sigma_tr^2*I) and testing channel N(0, sigma_te^2*I)"""
    var_tr = sigma_tr ** 2
    var_te = sigma_te ** 2
    # KL( P_tr || P_te ) for multivariate Gaussians
    # Using torch.log for consistency with bound calculation
    kl = 0.5 * d * (math.log(var_te / var_tr) + (var_tr / var_te) - 1)
    return kl

def compute_empirical_MI(W_samples, C_samples):
    """
    Computes the exact mutual information formula provided in the image:
    I(W; W^(l0)) = -0.5 * ln( |Sigma_joint| / (|Sigma_W| * |Sigma_C|) )
    """
    B = W_samples.size(0)
    
    # Center the samples
    W_centered = W_samples - W_samples.mean(dim=0)
    C_centered = C_samples - C_samples.mean(dim=0)
    
    # Create Joint samples
    joint = torch.cat([W_centered, C_centered], dim=1)
    
    # Compute empirical covariance matrices
    # Add a small jitter (1e-4) to the diagonal to ensure strictly positive-definite matrices
    Sigma_joint = (joint.T @ joint) / (B - 1) + 1e-4 * torch.eye(joint.size(1), device=W_samples.device)
    Sigma_W = (W_centered.T @ W_centered) / (B - 1) + 1e-4 * torch.eye(W_samples.size(1), device=W_samples.device)
    Sigma_C = (C_centered.T @ C_centered) / (B - 1) + 1e-4 * torch.eye(C_samples.size(1), device=W_samples.device)
    
    # Use slogdet for numerical stability (returns (sign, log_determinant))
    _, logdet_joint = torch.slogdet(Sigma_joint)
    _, logdet_W = torch.slogdet(Sigma_W)
    _, logdet_C = torch.slogdet(Sigma_C)
    
    # Log property: ln(a / (b*c)) = ln(a) - ln(b) - ln(c)
    mi = -0.5 * (logdet_joint - logdet_W - logdet_C)
    
    # Clamp to strictly non-negative values (empirical estimations can sometimes yield -1e-6)
    return torch.clamp(mi, min=0.0)

def compute_bound_rhs(model_kl, channel_kl, MI, n, k, epsilon, sigma_0, sigma_loss):
    """Computes the RHS of the wireless generalization error bound using torch ops for gradients."""
    # Ensure all inputs are converted to tensors to avoid math.sqrt warning
    device = model_kl.device
    k_tensor = torch.tensor(k, device=device)
    n_tensor = torch.tensor(n, device=device)
    eps_tensor = torch.tensor(epsilon, device=device)
    s0_tensor = torch.tensor(sigma_0, device=device)
    sl_tensor = torch.tensor(sigma_loss, device=device)
    
    term1 = torch.sqrt(2 * (s0_tensor**2) * (MI + channel_kl)) 
    term2 = (model_kl - torch.log(eps_tensor)) / k_tensor
    term3 = (k_tensor * (sl_tensor**2)) / (2 * n_tensor)
    return term1 + term2 + term3

# ==========================================
# 5. Training and Inference Functions
# ==========================================
def train_model(model, optimizer, X_train, y_train, mode="ERM"):
    """
    Trains the model based on the specified mode.
    Modes: 'ERM', 'PAC', 'BOUND'
    """
    model.train()
    n = X_train.size(0)
    
    # Calculate constant channel KL (does not require gradients)
    channel_kl = compute_channel_kl(model.channel_dim, SIGMA_TR, SIGMA_TE)
    channel_kl_tensor = torch.tensor(channel_kl, device=device)
    
    # Convert to DataLoader to ensure valid empirical batch sizes
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            # Forward pass returning dynamically sampled weights and channel noise
            logits, w_samples, c_samples = model(batch_x, SIGMA_TR)
            
            # Expected Empirical Risk
            empirical_risk = F.binary_cross_entropy_with_logits(logits, batch_y)
            
            model_kl = model.get_total_kl()
            mi_val = compute_empirical_MI(w_samples, c_samples)
            
            if mode == "ERM":
                loss = empirical_risk
            elif mode == "PAC":
                lambda_reg = 1.0 / n 
                loss = empirical_risk + lambda_reg * model_kl
            elif mode == "BOUND":
                rhs = compute_bound_rhs(model_kl, channel_kl_tensor, mi_val, n, K_PARAM, EPSILON, SIGMA_0, SIGMA_LOSS)
                loss = empirical_risk + rhs
            else:
                raise ValueError("Unknown mode")
                
            loss.backward()
            optimizer.step()

    # Final evaluation on training set
    model.eval()
    with torch.no_grad():
        logits, w_samples, c_samples = model(X_train, SIGMA_TR)
        final_erm = F.binary_cross_entropy_with_logits(logits, y_train).item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y_train).float().mean().item()
        final_mi = compute_empirical_MI(w_samples, c_samples).item()
        
    # Calculate Theoretical Bound explicitly for reporting
    final_model_kl = model.get_total_kl()
    bound_rhs_val = compute_bound_rhs(final_model_kl, channel_kl_tensor, torch.tensor(final_mi, device=device), n, K_PARAM, EPSILON, SIGMA_0, SIGMA_LOSS).item()
    
    return model, final_erm, acc, bound_rhs_val

def run_inference(model, X_test, y_test):
    """
    Inference process evaluating expected population risk under testing channel distribution.
    """
    model.eval()
    # To approximate the expected population risk, we take multiple Monte Carlo samples
    # over the posterior weights and the test channel noise.
    mc_samples = 50
    total_loss = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for _ in range(mc_samples):
            # Forward pass (using testing channel noise)
            logits, _, _ = model(X_test, SIGMA_TE)
            total_loss += F.binary_cross_entropy_with_logits(logits, y_test).item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_acc += (preds == y_test).float().mean().item()
            
    expected_pop_risk = total_loss / mc_samples
    expected_acc = total_acc / mc_samples
    
    return expected_pop_risk, expected_acc

# ==========================================
# 6. Main Execution Block
# ==========================================
def main():
    print(f"--- Starting Wireless Edge Learning Bound Verification ---")
    X_train, y_train, X_test, y_test = generate_synthetic_data(NUM_SAMPLES, INPUT_DIM)
    
    modes = ["ERM", "PAC", "BOUND"]
    results = {}
    
    for mode in modes:
        print(f"\nTraining with mode: {mode}...")
        model = WirelessEdgeNet(INPUT_DIM, CHANNEL_DIM, PRIOR_SIGMA).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        # Train
        trained_model, train_risk, train_acc, bound_rhs = train_model(model, optimizer, X_train, y_train, mode=mode)
        
        # Inference
        test_risk, test_acc = run_inference(trained_model, X_test, y_test)
        
        # Generalization Error
        empirical_gen_error = test_risk - train_risk
        
        # Save Weights
        weight_path = os.path.join(RESULTS_DIR, f"weights_{mode}.pth")
        torch.save(trained_model.state_dict(), weight_path)
        
        # Store results
        results[mode] = {
            "Train Risk (ERM)": train_risk,
            "Test Risk (Pop Risk)": test_risk,
            "Empirical Gen Error (Delta)": empirical_gen_error,
            "Theoretical Bound (RHS)": bound_rhs,
            "Train Acc": train_acc,
            "Test Acc": test_acc
        }
        
        print(f"Results for {mode}:")
        print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"  Empirical Gen Error: {empirical_gen_error:.4f} | Theoretical Bound: {bound_rhs:.4f}")

    # Save numeric results
    with open(os.path.join(RESULTS_DIR, "simulation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # --- Plotting ---
    labels = modes
    gen_errors = [results[m]["Empirical Gen Error (Delta)"] for m in modes]
    bounds = [results[m]["Theoretical Bound (RHS)"] for m in modes]
    
    x = np.arange(len(labels))
    width = 0.35

    # Plot 1: Gen Error vs Bound
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, gen_errors, width, label='Empirical Generalization Error ($\Delta$)')
    rects2 = ax.bar(x + width/2, bounds, width, label='Theoretical Bound (RHS)')

    ax.set_ylabel('Risk Difference')
    ax.set_title('Empirical Generalization Error vs. Theoretical Bound')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plot_path1 = os.path.join(RESULTS_DIR, "bound_comparison.png")
    plt.savefig(plot_path1, dpi=300)
    print(f"\nSaved Bound Comparison plot to {plot_path1}")
    
    # Plot 2: Accuracy Comparison
    train_accs = [results[m]["Train Acc"] for m in modes]
    test_accs = [results[m]["Test Acc"] for m in modes]
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    rects3 = ax2.bar(x - width/2, train_accs, width, label='Train Accuracy')
    rects4 = ax2.bar(x + width/2, test_accs, width, label='Test Accuracy')

    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison across Training Objectives')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    
    plt.tight_layout()
    plot_path2 = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
    plt.savefig(plot_path2, dpi=300)
    print(f"Saved Accuracy Comparison plot to {plot_path2}")
    
    print("\nSimulation Complete. All files saved successfully.")

if __name__ == "__main__":
    main()