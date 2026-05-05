import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set global seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

# ==========================================
# 1. Dataset Formulation (Instance Space Z)
# ==========================================
class SyntheticDataset(Dataset):
    """
    Synthesizes the dataset P_Z: 
    X ~ N(0, I_2)
    Y = a^T X + xi, where xi ~ N(0, sigma_xi^2)
    """
    def __init__(self, n_samples=1000, input_dim=2, noise_std=0.1, fixed_a=None, R_X=2.0, R_xi=0.2):
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.R_X = R_X
        self.R_xi = R_xi
        
        # Rejection sampling for X: ||X||_2 <= R_X
        X_list = []
        collected_X = 0
        while collected_X < n_samples:
            batch_X = torch.randn(n_samples, input_dim)
            norms_X = torch.norm(batch_X, p=2, dim=1)
            valid_X = batch_X[norms_X <= self.R_X]
            X_list.append(valid_X)
            collected_X += valid_X.size(0)
        self.X = torch.cat(X_list, dim=0)[:n_samples]
        
        # Fixed parameter vector 'a'
        if fixed_a is None:
            self.a = torch.ones(input_dim, 1) # Default a = [1, 1]^T
        else:
            self.a = torch.tensor(fixed_a, dtype=torch.float32).view(-1, 1)
            
        # Rejection sampling for xi: |xi| <= R_xi
        xi_list = []
        collected_xi = 0
        while collected_xi < n_samples:
            batch_xi = torch.randn(n_samples, 1) * noise_std
            valid_xi = batch_xi[torch.abs(batch_xi[:, 0]) <= self.R_xi]
            xi_list.append(valid_xi)
            collected_xi += valid_xi.size(0)
        self.xi = torch.cat(xi_list, dim=0)[:n_samples]
        
        # Y = a^T X + xi
        self.Y = torch.matmul(self.X, self.a) + self.xi

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def visualize_dataset(dataset, save_path=None):
    """
    Illustrates the synthetic dataset by plotting X (2D) and Y.
    Saves the image at 300 DPI if a save_path is provided.
    """
    X = dataset.X.numpy()
    Y = dataset.Y.numpy()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of the data points
    sc = ax.scatter(X[:, 0], X[:, 1], Y.flatten(), c=Y.flatten(), cmap='viridis', alpha=0.6)
    
    ax.set_xlabel('Feature $X_1$')
    ax.set_ylabel('Feature $X_2$')
    ax.set_zlabel('Label $Y$')
    ax.set_title(f'Visualization of Synthesized Dataset $P_Z$\n$Y = {dataset.a[0].item():.1f}X_1 + {dataset.a[1].item():.1f}X_2 + \\xi$')
    
    fig.colorbar(sc, ax=ax, label='Label value')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Dataset visualization saved to {save_path} at 300 DPI.")
    

# ==========================================
# 2. Bayesian Neural Network Components
# ==========================================
class BayesianLinear(nn.Module):
    """
    A single Bayesian Linear layer utilizing the reparameterization trick.
    Weights and biases are sampled from a Gaussian posterior.
    """
    def __init__(self, in_features, out_features, prior_var=1.0, max_norm=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var
        self.max_norm = max_norm
        
        # Variational parameters for the weight posterior: N(mu, sigma^2)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        
        # Variational parameters for the bias posterior
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        epsilon_w = torch.randn_like(weight_sigma)
        epsilon_b = torch.randn_like(bias_sigma)
        
        weight = self.weight_mu + weight_sigma * epsilon_w

        # Truncate weight if max_norm is specified to ensure ||W||_F <= R_W
        if self.max_norm is not None:
            w_norm = torch.norm(weight, p='fro')
            if w_norm > self.max_norm:
                weight = weight * (self.max_norm / w_norm)

        bias = self.bias_mu + bias_sigma * epsilon_b
        
        return nn.functional.linear(x, weight, bias)

    def kl_divergence(self):
        """Computes analytic KL divergence between posterior and prior N(0, prior_var)"""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        kl_w = 0.5 * torch.sum(
            (self.weight_mu**2 + weight_sigma**2) / self.prior_var 
            - 1 - 2 * torch.log(weight_sigma / math.sqrt(self.prior_var))
        )
        kl_b = 0.5 * torch.sum(
            (self.bias_mu**2 + bias_sigma**2) / self.prior_var 
            - 1 - 2 * torch.log(bias_sigma / math.sqrt(self.prior_var))
        )
        return kl_w + kl_b

# ==========================================
# 3. Wireless Channel Definition
# ==========================================
class EdgeWirelessChannel(nn.Module):
    """
    Models the continuous channel distortion layer.
    M_ij ~ N(delta_ij, alpha * ||f_1||_2^2)
    """
    def __init__(self, d, alpha=0.1):
        super().__init__()
        self.d = d
        self.alpha = alpha

    def forward(self, f1):
        batch_size = f1.size(0)
        f1_norm_sq = torch.sum(f1**2, dim=1, keepdim=True)
        sigma_m = torch.sqrt(self.alpha * f1_norm_sq).unsqueeze(-1)
        
        I_d = torch.eye(self.d, device=f1.device).unsqueeze(0).expand(batch_size, -1, -1)
        noise = torch.randn(batch_size, self.d, self.d, device=f1.device)
        
        M = I_d + sigma_m * noise
        f1_channelled = torch.bmm(M, f1.unsqueeze(2)).squeeze(2)
        
        return f1_channelled

    def expected_penalty(self, f1):
        """Analytically computes E[||M' - I_d||_F]"""
        f1_norm = torch.sqrt(torch.sum(f1**2, dim=1))
        sigma_m = math.sqrt(self.alpha) * f1_norm
        k = self.d ** 2
        chi_mean_factor = math.sqrt(2) * math.exp(math.lgamma((k + 1) / 2) - math.lgamma(k / 2))
        return (chi_mean_factor * sigma_m).mean()

# ==========================================
# 4. Augmented Neural Network
# ==========================================
class AugmentedBNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, alpha=0.1, R_W1=2.0, R_W3=2.0):
        super().__init__()
        self.layer1 = BayesianLinear(input_dim, hidden_dim, max_norm=R_W1)
        self.channel = EdgeWirelessChannel(hidden_dim, alpha)
        self.layer2 = BayesianLinear(hidden_dim, output_dim, max_norm=R_W3)

    def forward(self, x, simulate_channel=False):
        f1 = self.layer1(x)
        f1_out = self.channel(f1) if simulate_channel else f1
        out = self.layer2(f1_out)
        return out, f1

    def kl_divergence(self):
        return self.layer1.kl_divergence() + self.layer2.kl_divergence()

# ==========================================
# 5. Training and Testing Processes
# ==========================================

def train_process(objective="Derived_Bound", epochs=200, name='Linear_Gaussian_Continuous', simulate_channel=False, R_X=2.0, R_xi=0.2, R_W1=2.0, R_W3=2.0, R_loss=2.0, epsilon=0.01):
    """Runs the training phase and saves the model and empirical results."""
    print(f"--- [TRAIN] Strategy: {objective} ---")
    
    # Setup results directory
    res_path = os.path.join("./results", name)
    os.makedirs(res_path, exist_ok=True)
    
    # Constants and Truncation Bounds
    N_TRAIN = 1000
    BATCH_SIZE = 100
    LR = 0.01
    alpha = 0.1
    k_param = math.sqrt(N_TRAIN)
    
    # Calculate Exact Lipschitz K and sigma
    norm_a = math.sqrt(2.0) # For default a = [1, 1]^T
    R_Y = norm_a * R_X + R_xi
    K_lipschitz = 2 * (R_Y + R_W3 * R_W1 * R_X) * R_X * math.sqrt(R_W3**2 + R_W1**2)
    sigma_subg = (R_Y + R_W3 * R_W1 * R_X)**2 / 2

    set_seed(42)  # Ensure reproducibility
    
    train_data = SyntheticDataset(n_samples=N_TRAIN, R_X=R_X, R_xi=R_xi)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    # Visualize once and save the image with 300 DPI
    if objective == "ERM":
        img_save_path = os.path.join(res_path, "dataset_illustration.png")
        visualize_dataset(train_data, save_path=img_save_path)
        
    model = AugmentedBNN(input_dim=2, hidden_dim=4, output_dim=1, alpha=alpha, R_W1=R_W1, R_W3=R_W3)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    total_emp_risk = 0.0

    # record the maximum loss
    max_loss = 0.0
    max_grad_norm = 0.0
    
    model.train()
    for epoch in range(epochs):
        epoch_emp_risk = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds, f1 = model(batch_x, simulate_channel=simulate_channel)
            emp_risk = criterion(preds, batch_y)
            kl_div = model.kl_divergence()

            max_loss = max(max_loss, emp_risk.item())
            
            
            if objective == "ERM" or objective == "ERM_E2E":
                loss = emp_risk
            elif objective == "PAC_Bayes":
                loss = emp_risk + kl_div / k_param
            elif objective == "Derived_Bound":
                penalty = model.channel.expected_penalty(f1)
                kl_term = (kl_div - math.log(epsilon)) / k_param
                loss = emp_risk + K_lipschitz * penalty + kl_term
            
            loss.backward()
            
            # Record maximum gradient norm to approximate empirical Lipschitz constant
            # Using an effectively infinite max_norm to just retrieve the current norm value
            current_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf')).item()
            max_grad_norm = max(max_grad_norm, current_grad_norm)

            optimizer.step()
            epoch_emp_risk += emp_risk.item()
            
        total_emp_risk = epoch_emp_risk / len(train_loader)
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Emp Risk: {total_emp_risk:.4f}")

    # Save model and training metadata
    save_data = {
        'model_state_dict': model.state_dict(),
        'objective': objective,
        'final_empirical_risk': total_emp_risk,
        'hyperparams': {
            'N_TRAIN': N_TRAIN,
            'K_lipschitz': K_lipschitz,
            'k_param': k_param,
            'alpha': alpha,
            'R_X': R_X,
            'R_xi': R_xi,
            'R_W1': R_W1,
            'R_W3': R_W3,
            'epsilon': epsilon,
            # 'sigma_subg': sigma_subg
            'sigma_subg': max_loss ** 2 / 2
        }
    }
    torch.save(save_data, os.path.join(res_path, f"model_{objective}.pth"))
    print(f"Model and risk saved to {res_path}\n")
    # print(f"Maximum observed loss during training: {max_loss:.4f}")
    # print(f"Maximum observed gradient norm during training: {max_grad_norm:.4f}\n")

def test_process(objective="Derived_Bound", name='Linear_Gaussian_Continuous', simulate_channel=True):
    """Loads a saved model, calculates population risk and bound components."""
    print(f"--- [TEST] Strategy: {objective} ---")
    res_path = os.path.join("./results", name)
    file_path = os.path.join(res_path, f"model_{objective}.pth")
    
    if not os.path.exists(file_path):
        print(f"No saved model found for {objective}. Skipping.")
        return

    checkpoint = torch.load(file_path)
    hparams = checkpoint['hyperparams']
    
    model = AugmentedBNN(
        input_dim=2, hidden_dim=4, output_dim=1, 
        alpha=hparams['alpha'], R_W1=hparams['R_W1'], R_W3=hparams['R_W3']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    set_seed(999)  # Ensure reproducibility for testing
    
    test_data = SyntheticDataset(n_samples=1000, R_X=hparams['R_X'], R_xi=hparams['R_xi'])
    test_loader = DataLoader(test_data, batch_size=100)
    criterion = nn.MSELoss()
    
    # Calculate Population Risk (with channel noise)
    pop_risk_total = 0.0
    channel_penalty_total = 0.0
    mc_samples = 20
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_risk = 0
            for _ in range(mc_samples):
                preds, _ = model(batch_x, simulate_channel=simulate_channel)
                batch_risk += criterion(preds, batch_y).item()
            pop_risk_total += (batch_risk / mc_samples)
            
            # Channel penalty component for the bound
            _, f1_clean = model(batch_x, simulate_channel=False)
            channel_penalty_total += model.channel.expected_penalty(f1_clean).item()

    pop_risk = pop_risk_total / len(test_loader)
    avg_penalty = channel_penalty_total / len(test_loader)
    emp_risk = checkpoint['final_empirical_risk']
    kl_div = model.kl_divergence().item()
    
    # Exact Bound Components
    penalty_term = hparams['K_lipschitz'] * avg_penalty
    kl_term = (kl_div - math.log(hparams['epsilon'])) / hparams['k_param']
    variance_term = (hparams['k_param'] * hparams['sigma_subg']) / (2 * hparams['N_TRAIN'])
    
    total_upper_bound = emp_risk + penalty_term + kl_term + variance_term
    
    results = {
        'objective': objective,
        'population_risk': pop_risk,
        'empirical_risk': emp_risk,
        'total_upper_bound': total_upper_bound,
        'components': {
            'empirical_risk': emp_risk,
            'channel_penalty_term': penalty_term,
            'kl_complexity_term': kl_term,
            'variance_term': variance_term
        }
    }
    
    with open(os.path.join(res_path, f"results_{objective}.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Test Results for {objective}:")
    print(f"  Population Risk: {pop_risk:.4f}")
    print(f"  Upper Bound:     {total_upper_bound:.4f}")
    print(f"  Components:")
    print(f"    Risk:             {emp_risk:.4f}")
    print(f"    Channel Penalty:  {penalty_term:.4f}")
    print(f"    KL Term:          {kl_term:.4f}")
    print(f"    Variance Term:    {variance_term:.4f}")
    print(f"Results saved to {res_path}\n")

if __name__ == "__main__":
    strategies = ["ERM", "PAC_Bayes", "Derived_Bound", "ERM_E2E"]
    
    bounding_params = {
        "R_X": 2.0,
        "R_xi": 0.2,
        "R_W1": 4.0,
        "R_W3": 4.0,
        "R_loss": 2.0,
        "epsilon": 0.01
    }

    # Execute Training
    for strat in strategies:
        train_process(objective=strat, epochs=200, simulate_channel=(strat == "ERM_E2E"), **bounding_params)
        
    # Execute Testing
    for strat in strategies:
        test_process(objective=strat)