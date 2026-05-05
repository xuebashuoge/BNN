import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 1. HARDWARE SELECTION (Exact requested string)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. DATASET HANDLING
def get_dataset(n_samples=2000, n_features=20, noise=50.0):
    dataset_dir = os.environ.get("DATASET", "./dataset_cache")
    os.makedirs(dataset_dir, exist_ok=True)
    
    x_path = os.path.join(dataset_dir, "X_edge.npy")
    y_path = os.path.join(dataset_dir, "y_edge.npy")
    
    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"Loading dataset from {dataset_dir}")
        X = np.load(x_path)
        y = np.load(y_path)
    else:
        print(f"Generating and saving dataset to {dataset_dir}")
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
        # Normalize data to keep gradients stable
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = (y - y.mean()) / y.std()
        np.save(x_path, X)
        np.save(y_path, y)
        
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return (torch.tensor(X_train, dtype=torch.float32).to(device), 
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(X_test, dtype=torch.float32).to(device),
            torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device))

# 3. BAYESIAN NEURAL NETWORK COMPONENTS
class BayesianLinear(nn.Module):
    """
    Mean-Field Bayesian Linear Layer using the reparameterization trick.
    Prior Q_W is N(0, 1).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters for the posterior distribution P_{W|S}
        self.mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3.0, 0.1))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3.0, 0.1))

    @property
    def sigma(self):
        # Softplus to ensure standard deviation is strictly positive
        return F.softplus(self.rho) + 1e-6

    @property
    def bias_sigma(self):
        return F.softplus(self.bias_rho) + 1e-6

    def forward(self, x):
        # Reparameterization trick: w = mu + sigma * epsilon
        epsilon_w = torch.randn_like(self.mu)
        epsilon_b = torch.randn_like(self.bias_mu)
        
        w = self.mu + self.sigma * epsilon_w
        b = self.bias_mu + self.bias_sigma * epsilon_b
        
        return F.linear(x, w, b)

    def kl_prior(self):
        """Analytical KL Divergence between Posterior N(mu, sigma^2) and Prior N(0, 1)"""
        kl_w = 0.5 * (self.mu**2 + self.sigma**2 - 1 - 2 * torch.log(self.sigma)).sum()
        kl_b = 0.5 * (self.bias_mu**2 + self.bias_sigma**2 - 1 - 2 * torch.log(self.bias_sigma)).sum()
        return kl_w + kl_b

class ChannelLayer(nn.Module):
    """
    Non-learnable Wireless Channel Layer. W^{(l_0)} = (M, B).
    """
    def __init__(self, features, std_tr=1.0, std_te=1.5):
        super().__init__()
        self.features = features
        self.std_tr = std_tr
        self.std_te = std_te
        self.training_mode = True # True = use P_tr, False = use P_te

    def forward(self, x):
        std = self.std_tr if self.training_mode else self.std_te
        # Sample channel matrix M and bias B
        M = torch.randn(self.features, self.features, device=x.device) * std
        B = torch.randn(self.features, device=x.device) * std
        return F.linear(x, M, B)

    def kl_divergence_tr_te(self):
        """Analytical KL Divergence D(P_tr || P_te) for Gaussians"""
        dim = self.features * self.features + self.features
        var_tr = self.std_tr**2
        var_te = self.std_te**2
        # KL per dimension for zero-mean Gaussians: log(std_te/std_tr) + (var_tr)/(2*var_te) - 0.5
        kl_per_dim = np.log(self.std_te / self.std_tr) + (var_tr) / (2 * var_te) - 0.5
        return torch.tensor(kl_per_dim * dim, device=device, dtype=torch.float32)

class WirelessBNN(nn.Module):
    def __init__(self, in_features, hidden_dim=32, channel_dim=16, std_tr=1.0, std_te=1.5):
        super().__init__()
        # Transmitter
        self.tx1 = BayesianLinear(in_features, hidden_dim)
        self.tx2 = BayesianLinear(hidden_dim, channel_dim)
        
        # Wireless Channel
        self.channel = ChannelLayer(channel_dim, std_tr, std_te)
        
        # Receiver
        self.rx1 = BayesianLinear(channel_dim, hidden_dim)
        self.rx2 = BayesianLinear(hidden_dim, 1)
        
        self.bayesian_layers = [self.tx1, self.tx2, self.rx1, self.rx2]
        
        # EMA Buffers to track marginal P_{\tilde{W}|S}
        self.ema_decay = 0.95
        self.ema_params = {}

    def _init_ema(self):
        for i, layer in enumerate(self.bayesian_layers):
            self.ema_params[f'layer_{i}_mu'] = layer.mu.detach().clone()
            self.ema_params[f'layer_{i}_sigma'] = layer.sigma.detach().clone()
            self.ema_params[f'layer_{i}_b_mu'] = layer.bias_mu.detach().clone()
            self.ema_params[f'layer_{i}_b_sigma'] = layer.bias_sigma.detach().clone()

    def update_ema(self):
        with torch.no_grad():
            for i, layer in enumerate(self.bayesian_layers):
                self.ema_params[f'layer_{i}_mu'].mul_(self.ema_decay).add_(layer.mu.detach() * (1 - self.ema_decay))
                self.ema_params[f'layer_{i}_sigma'].mul_(self.ema_decay).add_(layer.sigma.detach() * (1 - self.ema_decay))
                self.ema_params[f'layer_{i}_b_mu'].mul_(self.ema_decay).add_(layer.bias_mu.detach() * (1 - self.ema_decay))
                self.ema_params[f'layer_{i}_b_sigma'].mul_(self.ema_decay).add_(layer.bias_sigma.detach() * (1 - self.ema_decay))

    def get_marginal_kl(self):
        """
        Approximate D(P_{W|S} || Q_W) using the EMA parameters (moment matched marginal).
        Prior Q_W is N(0, 1).
        """
        kl_marg = 0.0
        for i, layer in enumerate(self.bayesian_layers):
            ema_mu = self.ema_params[f'layer_{i}_mu']
            ema_sigma = self.ema_params[f'layer_{i}_sigma']
            kl_marg += 0.5 * (ema_mu**2 + ema_sigma**2 - 1 - 2 * torch.log(ema_sigma)).sum()
            
            ema_b_mu = self.ema_params[f'layer_{i}_b_mu']
            ema_b_sigma = self.ema_params[f'layer_{i}_b_sigma']
            kl_marg += 0.5 * (ema_b_mu**2 + ema_b_sigma**2 - 1 - 2 * torch.log(ema_b_sigma)).sum()
        return kl_marg

    def get_expected_conditional_kl(self):
        """
        Calculates E_{w_0}[D(P_{W|W_0, S} || Q_W)].
        Evaluated using the current batch's posterior against Prior N(0, 1).
        """
        return sum(layer.kl_prior() for layer in self.bayesian_layers)

    def forward(self, x):
        x = F.relu(self.tx1(x))
        x = self.tx2(x)
        x = self.channel(x) # Channel mismatch happens here
        x = F.relu(self.rx1(x))
        x = self.rx2(x)
        return x

# 4. TRAINING ENGINES
def train_model(mode, X_train, y_train, X_test, y_test, epochs=300):
    n = X_train.shape[0]
    in_features = X_train.shape[1]
    
    # Constants for the bound simulation (Assumed 1.0 for stability)
    sigma_loss = 1.0   
    sigma_0 = 1.0      
    epsilon = 0.05
    log_term = np.log(np.sqrt(n) / epsilon)

    model = WirelessBNN(in_features=in_features).to(device)
    model._init_ema() # Initialize EMA buffers
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    history = {
        'emp_risk': [], 
        'pop_risk': [], 
        'bound': []
    }

    for epoch in range(epochs):
        model.train()
        model.channel.training_mode = True # P_tr
        optimizer.zero_grad()
        
        # 1. Forward Pass (Empirical Risk)
        preds = model(X_train)
        emp_risk = F.mse_loss(preds, y_train)
        
        # 2. Compute Information-Theoretic Terms
        kl_cond = model.get_expected_conditional_kl()
        kl_marg = model.get_marginal_kl()
        kl_channel = model.channel.kl_divergence_tr_te()
        
        # Identity: I(W; W_0 | S) = E[D(P_{W|W_0,S} || Q)] - D(P_{W|S} || Q)
        # ReLU applied to handle transient negative values caused by EMA lag
        mi = torch.relu(kl_cond - kl_marg)
        
        # Standard PAC-Bayes Term (uses Marginal KL)
        pac_bayes_penalty = torch.sqrt((2 * sigma_loss**2) / (n - 1) * (kl_marg + log_term) + 1e-8)
        
        # Proposed Wireless Bound Term (uses MI and Expected Conditional KL)
        # Note: The theoretical formula uses kl_cond inside the second square root.
        proposed_term_1 = torch.sqrt(2 * sigma_0**2 * (mi + kl_channel) + 1e-8)
        proposed_term_2 = torch.sqrt((2 * sigma_loss**2) / (n - 1) * (kl_cond + log_term) + 1e-8)
        wireless_penalty = proposed_term_1 + proposed_term_2
        
        # 3. Apply Loss based on Mode
        if mode == 'ERM':
            loss = emp_risk
        elif mode == 'PAC-Bayes':
            # Scale by 1/n to match average loss scale, heavily used in BNNs
            loss = emp_risk + pac_bayes_penalty / n 
        elif mode == 'Proposed':
            loss = emp_risk + wireless_penalty / n
            
        loss.backward()
        optimizer.step()
        model.update_ema() # Update marginals P_{W|S}
        
        # 4. Evaluation (Population Risk under P_te)
        if epoch % 5 == 0:
            model.eval()
            model.channel.training_mode = False # Simulate deployment mismatch P_te
            with torch.no_grad():
                test_preds = model(X_test)
                pop_risk = F.mse_loss(test_preds, y_test)
                
                # Calculate theoretical bound mathematically
                kl_cond_eval = model.get_expected_conditional_kl()
                kl_marg_eval = model.get_marginal_kl()
                mi_eval = torch.relu(kl_cond_eval - kl_marg_eval)
                
                delta_bound = (torch.sqrt(2 * sigma_0**2 * (mi_eval + kl_channel)) + 
                               torch.sqrt((2 * sigma_loss**2)/(n - 1) * (kl_cond_eval + log_term)))
                total_bound = emp_risk + delta_bound
                
                history['emp_risk'].append(emp_risk.item())
                history['pop_risk'].append(pop_risk.item())
                history['bound'].append(total_bound.item())

            print(f"[{mode}] Epoch {epoch} | Emp Risk: {emp_risk.item():.4f} | Pop Risk: {pop_risk.item():.4f} | Bound: {total_bound.item():.4f} | MI: {mi_eval.item():.4f} | KL_channel: {kl_channel.item():.4f} | KL_cond: {kl_cond_eval.item():.4f} | KL_marg: {kl_marg_eval.item():.4f}")

    return history

# 5. EXECUTION & PLOTTING
def main():
    X_train, y_train, X_test, y_test = get_dataset()
    
    modes = ['ERM', 'PAC-Bayes', 'Proposed']
    results = {}
    
    for mode in modes:
        print(f"--- Training {mode} ---")
        results[mode] = train_model(mode, X_train, y_train, X_test, y_test)
        
    # Plotting
    epochs_range = np.arange(0, 300, 5)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, mode in enumerate(modes):
        ax = axes[i]
        ax.plot(epochs_range, results[mode]['emp_risk'], label='Empirical Risk (P_tr)', color='blue')
        ax.plot(epochs_range, results[mode]['pop_risk'], label='Expected Pop Risk (P_te)', color='red')
        ax.plot(epochs_range, results[mode]['bound'], label='Theoretical Bound', color='green', linestyle='--')
        
        ax.set_title(f'{mode} Training Behavior')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss / Risk')
        ax.set_ylim(0, 3) # Constrain for readability due to initial noisy bounds
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('wireless_edge_bounds.png')
    print("Simulation complete. Plot saved as 'wireless_edge_bounds.png'.")
    plt.show()

if __name__ == "__main__":
    main()