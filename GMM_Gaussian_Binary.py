import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set global seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ==========================================
# 1. Dataset Formulation (Instance Space Z)
# ==========================================
class GMMBinaryDataset(Dataset):
    """
    Synthesizes the dataset P_Z (Binary classification with Gaussian Mixture):
    Y in {-1, +1} with equal probability.
    X | Y=+1 ~ N(mu, Sigma)
    X | Y=-1 ~ N(-mu, Sigma)
    """
    def __init__(self, n_samples=1000, mu_val=1.5, sigma_val=0.5):
        self.n_samples = n_samples
        half = n_samples // 2
        
        # Define mu and Sigma
        mu = torch.tensor([mu_val, mu_val], dtype=torch.float32)
        Sigma = torch.eye(2) * sigma_val
        
        # Sample Class +1
        dist_pos = torch.distributions.MultivariateNormal(mu, Sigma)
        X_pos = dist_pos.sample((half,))
        Y_pos = torch.ones(half, 1)
        
        # Sample Class -1
        dist_neg = torch.distributions.MultivariateNormal(-mu, Sigma)
        X_neg = dist_neg.sample((n_samples - half,))
        Y_neg = -torch.ones(n_samples - half, 1)
        
        # Concatenate and shuffle
        X_all = torch.cat([X_pos, X_neg], dim=0)
        Y_all = torch.cat([Y_pos, Y_neg], dim=0)
        
        indices = torch.randperm(n_samples)
        self.X = X_all[indices]
        self.Y = Y_all[indices]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def visualize_dataset(dataset, save_path=None):
    """
    Illustrates the synthetic GMM dataset by plotting X (2D) and coloring by Y.
    Saves the image at 300 DPI if a save_path is provided.
    """
    X = dataset.X.numpy()
    Y = dataset.Y.numpy().flatten()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c='blue', label='Class +1', alpha=0.6, edgecolors='w')
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], c='red', label='Class -1', alpha=0.6, edgecolors='w')
    
    plt.xlabel('Feature $X_1$')
    plt.ylabel('Feature $X_2$')
    plt.title('Gaussian Mixture Binary Dataset ($P_Z$)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Dataset visualization saved to {save_path} at 300 DPI.")
    plt.close()

# ==========================================
# 2. Bayesian Neural Network Components
# ==========================================
class BayesianLinear(nn.Module):
    """
    A single Bayesian Linear layer utilizing the reparameterization trick.
    Weights and biases are sampled from a Gaussian posterior.
    """
    def __init__(self, in_features, out_features, prior_var=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var
        
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
class BernoulliDiagonalChannel(nn.Module):
    """
    Models the binary diagonal channel.
    f^{(1)}_out = M f^{(1)}
    M = diag(M_1, ..., M_d)
    P(M_i = 1 | f_i) = exp(-beta * f_i)
    """
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta

    def forward(self, f1):
        # f1 is expected to be non-negative due to ReLU
        probs = torch.exp(-self.beta * f1)
        
        if self.training:
            # Straight-Through Estimator (STE) for discrete sampling during ERM_E2E training
            sample = torch.bernoulli(probs)
            M = probs + (sample - probs).detach()
        else:
            M = torch.bernoulli(probs)
            
        return M * f1

    def expected_penalty(self, f1):
        """
        Analytically bounding E[||M' - I_d||_F].
        ||M' - I_d||_F = sqrt(sum_{i=1}^d (1 - M_i)^2) = sqrt(sum_{i=1}^d (1 - M_i))
        Using Jensen's Inequality: E[sqrt(X)] <= sqrt(E[X])
        E[X] = sum(1 - P(M_i=1)) = sum(1 - exp(-beta * f_i))
        """
        probs = torch.exp(-self.beta * f1)
        expected_errors = torch.sum(1.0 - probs, dim=1) # Expected number of flipped bits
        # Upper bound of the expected Frobenius norm penalty
        penalty_upper_bound = torch.sqrt(expected_errors + 1e-8)
        return penalty_upper_bound.mean()

# ==========================================
# 4. Augmented Neural Network
# ==========================================
class AugmentedBNN(nn.Module):
    """
    L=2 Bayesian Neural Network with ReLU activation and intermediate Wireless Channel.
    """
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=1, beta=0.5):
        super().__init__()
        self.layer1 = BayesianLinear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.channel = BernoulliDiagonalChannel(beta)
        self.layer2 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x, simulate_channel=False):
        z1 = self.layer1(x)
        f1 = self.act(z1) # Hidden feature representation
        
        if simulate_channel:
            f1_out = self.channel(f1)
        else:
            f1_out = f1
            
        out = self.layer2(f1_out)
        return out, f1

    def kl_divergence(self):
        return self.layer1.kl_divergence() + self.layer2.kl_divergence()

# ==========================================
# 5. Training and Testing Processes
# ==========================================
def train_process(objective="Derived_Bound", epochs=150, name='GMM_Gaussian_Binary', simulate_channel=False):
    """Runs the training phase and saves the model and empirical results."""
    print(f"--- [TRAIN] Strategy: {objective} ---")
    
    # Setup results directory
    res_path = os.path.join("./results", name)
    os.makedirs(res_path, exist_ok=True)
    
    # Hyperparameters
    N_TRAIN = 1000
    BATCH_SIZE = 100
    LR = 0.01
    K_lipschitz = 0.5
    k_param = 100.0
    beta = 0.5

    set_seed(42)  # Ensure reproducibility
    
    train_data = GMMBinaryDataset(n_samples=N_TRAIN)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    # Visualize once and save the image with 300 DPI
    if objective == "ERM":
        img_save_path = os.path.join(res_path, "dataset_illustration.png")
        visualize_dataset(train_data, save_path=img_save_path)
        
    model = AugmentedBNN(input_dim=2, hidden_dim=8, output_dim=1, beta=beta)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    total_emp_risk = 0.0
    
    model.train()
    for epoch in range(epochs):
        epoch_emp_risk = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            preds, f1 = model(batch_x, simulate_channel=simulate_channel)
            emp_risk = criterion(preds, batch_y)
            kl_div = model.kl_divergence()
            
            if objective == "ERM" or objective == "ERM_E2E":
                loss = emp_risk
            elif objective == "PAC_Bayes":
                loss = emp_risk + kl_div / (k_param * N_TRAIN)
            elif objective == "Derived_Bound":
                penalty = model.channel.expected_penalty(f1)
                loss = emp_risk + K_lipschitz * penalty + kl_div / (k_param * N_TRAIN)
            
            loss.backward()
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
            'beta': beta
        }
    }
    torch.save(save_data, os.path.join(res_path, f"model_{objective}.pth"))
    print(f"Model and risk saved to {res_path}\n")

def test_process(objective="Derived_Bound", name='GMM_Gaussian_Binary', simulate_channel=True):
    """Loads a saved model, calculates population risk and bound components."""
    print(f"--- [TEST] Strategy: {objective} ---")
    res_path = os.path.join("./results", name)
    file_path = os.path.join(res_path, f"model_{objective}.pth")
    
    if not os.path.exists(file_path):
        print(f"No saved model found for {objective}. Skipping.")
        return

    checkpoint = torch.load(file_path)
    hparams = checkpoint['hyperparams']
    
    model = AugmentedBNN(input_dim=2, hidden_dim=8, output_dim=1, beta=hparams['beta'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    set_seed(999)  # Ensure reproducibility for testing
    
    test_data = GMMBinaryDataset(n_samples=2000)
    test_loader = DataLoader(test_data, batch_size=200)
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
            
            # Channel penalty component for the theoretical bound (computed on clean features)
            _, f1_clean = model(batch_x, simulate_channel=False)
            channel_penalty_total += model.channel.expected_penalty(f1_clean).item()

    pop_risk = pop_risk_total / len(test_loader)
    avg_penalty = channel_penalty_total / len(test_loader)
    
    emp_risk = checkpoint['final_empirical_risk']
    kl_div = model.kl_divergence().item()
    
    # Bound Components Computation
    penalty_term = hparams['K_lipschitz'] * avg_penalty
    complexity_term = kl_div / (hparams['k_param'] * hparams['N_TRAIN'])
    
    total_upper_bound = emp_risk + penalty_term + complexity_term
    
    results = {
        'objective': objective,
        'population_risk': pop_risk,
        'total_upper_bound': total_upper_bound,
        'components': {
            'empirical_risk': emp_risk,
            'channel_penalty_term': penalty_term,
            'kl_complexity_term': complexity_term
        }
    }
    
    with open(os.path.join(res_path, f"results_{objective}.json"), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Test Results for {objective}:")
    print(f"  Population Risk: {pop_risk:.4f}")
    print(f"  Upper Bound:     {total_upper_bound:.4f}")
    print(f"  Components:      Risk: {emp_risk:.4f}, Channel Penalty: {penalty_term:.4f}, KL: {complexity_term:.4f}")
    print(f"Results saved to {res_path}\n")

if __name__ == "__main__":
    strategies = ["ERM", "PAC_Bayes", "Derived_Bound", "ERM_E2E"]
    experiment_name = "GMM_Gaussian_Binary"
    
    # Execute Training Phase
    for strat in strategies:
        # We simulate the channel explicitly during the ERM_E2E robust training objective
        train_process(objective=strat, epochs=200, name=experiment_name, simulate_channel=(strat == "ERM_E2E"))
        
    # Execute Testing Phase (Simulating the Edge inference noise)
    for strat in strategies:
        test_process(objective=strat, name=experiment_name, simulate_channel=True)