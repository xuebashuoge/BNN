import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # Enable MPS fallback for compatibility on M-series Macs
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Device and Environment Configuration
# ==========================================
# EXACT requirement: M-series Mac support
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset path management
DATASET_PATH = os.environ.get('DATASET', './data') # Fallback to ./data if not set
os.makedirs(DATASET_PATH, exist_ok=True)
dataset_file = os.path.join(DATASET_PATH, 'synthetic_wireless_data.pt')

# ==========================================
# 2. Dataset Generation
# ==========================================
def get_dataset(file_path):
    """
    Generates a high-dimensional, noisy classification dataset (overlapping Gaussians).
    Saves to the specified path to avoid re-generation.
    """
    if os.path.exists(file_path):
        print(f"Loading existing dataset from {file_path}")
        data = torch.load(file_path)
        X, Y = data['X'], data['Y']
    else:
        print(f"Generating new dataset and saving to {file_path}")
        # 12 features to keep the network small enough for stable Covariance determinant math
        X_np, Y_np = make_classification(
            n_samples=2500, n_features=12, n_informative=10, n_redundant=2,
            n_classes=2, class_sep=0.5, flip_y=0.1, random_state=42 # Noisy labels & overlap
        )
        X = torch.tensor(X_np, dtype=torch.float32)
        Y = torch.tensor(Y_np, dtype=torch.long)
        torch.save({'X': X, 'Y': Y}, file_path)
    return X, Y

X, Y = get_dataset(dataset_file)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)

# ==========================================
# 3. Model Components
# ==========================================
class BayesianLinear(nn.Module):
    """
    Learnable BNN layer using the reparameterization trick.
    Calculates posterior P_{\tilde{W}} and its KL divergence from prior Q_{\tilde{W}} = N(0, lambda*I).
    """
    def __init__(self, in_features, out_features, prior_var=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var
        
        # Learnable parameters for the posterior distribution q(w) = N(mu, sigma^2)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

    @property
    def weight_sigma(self):
        return torch.log1p(torch.exp(self.weight_rho)) # Softplus for positivity

    @property
    def bias_sigma(self):
        return torch.log1p(torch.exp(self.bias_rho))

    def forward(self, x, sample=True):
        if sample:
            # Reparameterization trick: w = mu + sigma * epsilon
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + self.weight_sigma * weight_eps
            bias = self.bias_mu + self.bias_sigma * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def compute_kl(self):
        """Analytic KL divergence between Posterior N(mu, sigma^2) and Prior N(0, prior_var)"""
        w_kl = 0.5 * (self.weight_sigma**2 / self.prior_var + self.weight_mu**2 / self.prior_var 
                      - 1 - 2 * torch.log(self.weight_sigma) + math.log(self.prior_var))
        b_kl = 0.5 * (self.bias_sigma**2 / self.prior_var + self.bias_mu**2 / self.prior_var 
                      - 1 - 2 * torch.log(self.bias_sigma) + math.log(self.prior_var))
        return w_kl.sum() + b_kl.sum()
    
    def sample_weights(self, num_samples):
        """Returns flattened weight samples for Mutual Information computation"""
        eps_w = torch.randn(num_samples, self.out_features, self.in_features, device=device)
        eps_b = torch.randn(num_samples, self.out_features, device=device)
        
        w_samples = self.weight_mu.unsqueeze(0) + self.weight_sigma.unsqueeze(0) * eps_w
        b_samples = self.bias_mu.unsqueeze(0) + self.bias_sigma.unsqueeze(0) * eps_b
        
        # Flatten and concatenate
        w_flat = w_samples.view(num_samples, -1)
        b_flat = b_samples.view(num_samples, -1)
        return torch.cat([w_flat, b_flat], dim=1)

class ChannelLayer(nn.Module):
    """
    Non-learnable Channel Layer (l_0). Weights are sampled from P_tr during training
    and P_te during inference.
    """
    def __init__(self, features, mu_tr=0.0, std_tr=0.1, mu_te=0.5, std_te=0.3):
        super().__init__()
        self.features = features
        self.mu_tr, self.std_tr = mu_tr, std_tr
        self.mu_te, self.std_te = mu_te, std_te
        
    def forward(self, x, mode='train'):
        mu = self.mu_tr if mode == 'train' else self.mu_te
        std = self.std_tr if mode == 'train' else self.std_te
        
        # Sample channel matrix M and bias B
        M = torch.randn(self.features, self.features, device=device) * std + mu
        B = torch.randn(self.features, device=device) * std + mu
        return F.linear(x, M, B)

    def compute_channel_kl(self):
        """
        Analytic KL divergence D(P_tr || P_te) for the channel weights.
        Assumes independent Gaussians for all elements in M and B.
        """
        dim = self.features * self.features + self.features
        var_tr, var_te = self.std_tr**2, self.std_te**2
        kl = 0.5 * dim * (math.log(var_te / var_tr) + (var_tr + (self.mu_tr - self.mu_te)**2) / var_te - 1)
        return torch.tensor(kl, device=device)

    def sample_weights(self, num_samples, mode='train'):
        mu = self.mu_tr if mode == 'train' else self.mu_te
        std = self.std_tr if mode == 'train' else self.std_te
        dim = self.features * self.features + self.features
        return torch.randn(num_samples, dim, device=device) * std + mu

class EdgeNetwork(nn.Module):
    """Full 3-Layer architecture: Tx BNN -> Channel -> Rx BNN"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.tx_layer = BayesianLinear(in_dim, hidden_dim)
        self.channel = ChannelLayer(hidden_dim)
        self.rx_layer = BayesianLinear(hidden_dim, out_dim)
        
    def forward(self, x, mode='train', sample=True):
        x = F.relu(self.tx_layer(x, sample))
        x = F.relu(self.channel(x, mode))
        x = self.rx_layer(x, sample)
        return x

# ==========================================
# 4. Theoretical Bound Components
# ==========================================
def compute_mutual_information_surrogate(model, num_samples=300):
    """
    ASSUMPTION: We approximate I(W_tilde; W^(l_0) | S) by drawing joint samples
    of the BNN weights and Channel weights, treating them as a joint Multivariate Gaussian.
    To prevent rank deficiency in empirical covariance, we keep weight dimensions small 
    and draw more samples than the total number of parameters.
    """
    # Sample weights
    tx_w = model.tx_layer.sample_weights(num_samples)
    rx_w = model.rx_layer.sample_weights(num_samples)
    w_tilde = torch.cat([tx_w, rx_w], dim=1) # Learnable parameters
    w_ch = model.channel.sample_weights(num_samples, mode='train') # Channel parameters
    
    # Center the samples
    w_tilde = w_tilde - w_tilde.mean(dim=0)
    w_ch = w_ch - w_ch.mean(dim=0)
    
    # Empirical Covariance Matrices (with jitter for numerical stability)
    jitter = 1e-3
    C_tt = (w_tilde.T @ w_tilde) / (num_samples - 1) + jitter * torch.eye(w_tilde.size(1), device=device)
    C_cc = (w_ch.T @ w_ch) / (num_samples - 1) + jitter * torch.eye(w_ch.size(1), device=device)
    C_tc = (w_tilde.T @ w_ch) / (num_samples - 1)
    C_ct = C_tc.T
    
    # Joint Covariance Matrix
    C_joint_top = torch.cat([C_tt, C_tc], dim=1)
    C_joint_bot = torch.cat([C_ct, C_cc], dim=1)
    C_joint = torch.cat([C_joint_top, C_joint_bot], dim=0)
    
    # Calculate log determinants
    _, logdet_joint = torch.linalg.slogdet(C_joint)
    _, logdet_tt = torch.linalg.slogdet(C_tt)
    _, logdet_cc = torch.linalg.slogdet(C_cc)
    
    # Mutual Information: 0.5 * log( |C_tt|*|C_cc| / |C_joint| )
    mi = 0.5 * (logdet_tt + logdet_cc - logdet_joint)
    
    # Ensure it's non-negative (numerical artifacts might push it slightly below 0)
    return F.relu(mi)

def calculate_theoretical_bound(model, n, k=1.0, sigma=1.0, sigma_0=1.0, epsilon=0.05):
    """
    Calculates the exact high-probability upper bound formula provided:
    Delta <= sqrt( 2 * sigma_0^2 * (I + D_channel) ) + (KL_w - log(eps))/k + (k * sigma^2)/(2n)
    """
    kl_w = model.tx_layer.compute_kl() + model.rx_layer.compute_kl()
    d_channel = model.channel.compute_channel_kl()
    mi = compute_mutual_information_surrogate(model)
    
    term1 = torch.sqrt(2 * sigma_0**2 * (mi + d_channel) + 1e-8) # 1e-8 for stable sqrt gradient
    term2 = (kl_w - math.log(epsilon)) / k
    term3 = (k * sigma**2) / (2 * n)
    
    bound = term1 + term2 + term3
    return bound, kl_w, d_channel, mi, term1, term2, term3

# ==========================================
# 5. Training Loop
# ==========================================
def train_model(model, X_tr, Y_tr, X_te, Y_te, epochs=100, use_regularization='ERM'):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    n = X_tr.shape[0]
    
    history = {'emp_risk': [], 'pop_risk': [], 'delta': [], 'bound': []}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 1. Empirical Risk on Training Data (Channel ~ P_tr)
        logits_tr = model(X_tr, mode='train', sample=True)
        loss_erm = criterion(logits_tr, Y_tr)
        
        # 2. Bound Calculation
        bound, kl_w, d_ch, mi, term1, term2, term3 = calculate_theoretical_bound(model, n)
        
        # 3. Optimization Step
        # If regularizing, scale the bound by a factor to prevent it from overwhelming the CE loss
        if use_regularization.lower() == 'bound':
            total_loss = loss_erm + 0.01 * bound
        elif use_regularization.lower() == 'pac-bayes':
            total_loss = loss_erm + 0.01 * kl_w
        else:
            total_loss = loss_erm
        total_loss.backward()
        optimizer.step()
        
        # 4. Evaluation (Population Risk on Test Data, Channel ~ P_te)
        model.eval()
        with torch.no_grad():
            logits_te = model(X_te, mode='test', sample=False) # Freeze BNN weights to mu, Channel ~ P_te
            loss_pop = criterion(logits_te, Y_te)
            
            delta = loss_pop - loss_erm
            
            history['emp_risk'].append(loss_erm.item())
            history['pop_risk'].append(loss_pop.item())
            history['delta'].append(delta.item())
            history['bound'].append(bound.item())
            
        if epoch % 20 == 0:
            reg_status = use_regularization
            print(f"[{reg_status}] Epoch {epoch} | Emp Risk: {loss_erm.item():.4f} | Pop Risk: {loss_pop.item():.4f} | Delta: {delta.item():.4f} | Bound: {bound.item():.4f} (term1: {term1.item():.4f}, term2: {term2.item():.4f}, term3: {term3:.4f})")
            
    return history

# ==========================================
# 6. Execution & Plotting
# ==========================================
print("\n--- Phase A: Training with Standard ERM ---")
# Keep hidden dim strictly small (8) to ensure empirical covariance matrices are invertible!
model_erm = EdgeNetwork(in_dim=12, hidden_dim=8, out_dim=2).to(device)
history_erm = train_model(model_erm, X_train, Y_train, X_test, Y_test, epochs=1000, use_regularization='ERM')

print("\n--- Phase B: Training with PAC-Bayes Regularization ---")
model_pac_bayes = EdgeNetwork(in_dim=12, hidden_dim=8, out_dim=2).to(device)
history_pac_bayes = train_model(model_pac_bayes, X_train, Y_train, X_test, Y_test, epochs=1000, use_regularization='PAC-BAYES')

print("\n--- Phase C: Training with Bound Regularization ---")
model_bound = EdgeNetwork(in_dim=12, hidden_dim=8, out_dim=2).to(device)
history_bound = train_model(model_bound, X_train, Y_train, X_test, Y_test, epochs=1000, use_regularization='BOUND')
# Plotting the results
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Verifying Delta <= Bound (ERM Case)
epochs_range = range(len(history_erm['delta']))
axs[0].plot(epochs_range, history_erm['delta'], label='Empirical Delta ($\Delta$)', color='blue')
axs[0].plot(epochs_range, history_erm['bound'], label='Theoretical Bound', color='red', linestyle='--')
axs[0].set_title('(a) Wireless Gen. Error vs. Theoretical Bound (ERM)')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss Difference')
axs[0].legend()
axs[0].grid(True)

# Plot 2: Population Risk Comparison (ERM vs Regularized)
axs[1].plot(epochs_range, history_erm['pop_risk'], label='Population Risk (Standard ERM)', color='blue')
axs[1].plot(epochs_range, history_pac_bayes['pop_risk'], label='Population Risk (PAC-Bayes Regularized)', color='orange')
axs[1].plot(epochs_range, history_bound['pop_risk'], label='Population Risk (Bound Regularized)', color='green')
axs[1].set_title('(b) Expected Population Risk Reduction')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Test Loss (under $P_{te}$ channel)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('wireless_edge_results.png')
print("\nSimulation complete. Results saved to 'wireless_edge_results.png'.")