import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================
# 1. HARDWARE & PATH CONFIGURATION (CRITICAL)
# ==========================================
# Exact device string requested for Mac M-series fallback
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Exact dataset path logic requested
data_path = os.environ.get('DATASET', './data')
os.makedirs(data_path, exist_ok=True)
print(f"Dataset path set to: {data_path}")

# ==========================================
# 2. HYPERPARAMETERS & CHANNEL DISTRIBUTIONS
# ==========================================
K_ENSEMBLE = 10        # Number of BNNs in the ensemble
HIDDEN_DIM = 128       # INCREASED: Give ERM enough capacity to overfit the channel
IN_DIM = 20
OUT_DIM = 2
BATCH_SIZE = 64        # Reduced batch size for noisier gradients
EPOCHS = 150           # Increased epochs to ensure ERM fully memorizes
PRIOR_LAMBDA = 1.0     # Variance of isotropic Gaussian prior
EPSILON = 0.05         # PAC-Bayes confidence parameter
SIGMA_SQ = 1.0         # Assumed sub-Gaussian parameter
SIGMA_0_SQ = 1.0       # Assumed sub-Gaussian parameter
REG_COEFF = 0.05       # Adjusted: Give the regularizer a slightly stronger voice

# Channel Distributions (Train) - DECEPTIVELY CLEAN
# ERM will become overconfident and build fragile decision boundaries.
MU_M_TR, STD_M_TR = 1.0, 0.01  # Almost no fading variation
MU_B_TR, STD_B_TR = 0.0, 0.01  # Almost no AWGN

# Channel Distributions (Inference/Test) - HARSH REALITY
# ERM's fragile boundaries will fail here. Bayesian smoothing should win.
MU_M_TE, STD_M_TE = 0.5, 0.5   # Severe fading
MU_B_TE, STD_B_TE = 0.0, 1.5   # Heavy noise

def kl_gaussian_1d(mu1, std1, mu2, std2):
    """Closed form KL D(N_1 || N_2)"""
    return np.log(std2/std1) + (std1**2 + (mu1 - mu2)**2) / (2 * std2**2) - 0.5

# Closed form KL between Train and Test channels
kl_ch_m = HIDDEN_DIM * kl_gaussian_1d(MU_M_TR, STD_M_TR, MU_M_TE, STD_M_TE)
kl_ch_b = HIDDEN_DIM * kl_gaussian_1d(MU_B_TR, STD_B_TR, MU_B_TE, STD_B_TE)
KL_CH_TOTAL = kl_ch_m + kl_ch_b
print(f"Constant D(P_tr || P_te) = {KL_CH_TOTAL:.4f}")


# ==========================================
# 3. DATA GENERATION
# ==========================================
def get_dataloaders(n_samples=300): # DECREASED: Less data makes ERM overfit faster
    # Significant label noise and overlapping classes to prevent trivial ERM collapse
    X, y = make_classification(
        n_samples=n_samples, n_features=IN_DIM, n_informative=10, 
        n_classes=OUT_DIM, flip_y=0.15, random_state=42, class_sep=0.5
    )

    visualize_and_save_dataset(X, y, filename=os.path.join(data_path, "dataset_viz.png"), title="Synthetic Dataset Visualization")

    # Convert to float32 for PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader, n_samples

def visualize_and_save_dataset(X, y, filename="dataset_viz.png", title="Dataset Visualization"):
    """
    Reduces dimensionality to 2D using PCA and saves a scatter plot.
    """
    # 1. Reduce dimensions to 2D
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    # 2. Setup the plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_reduced[:, 0], 
        X_reduced[:, 1], 
        c=y, 
        cmap='viridis', 
        edgecolors='k', 
        alpha=0.7
    )
    
    # 3. Add aesthetics
    plt.colorbar(scatter, label='Class Label')
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Save and close
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close()
    print(f"Visualization saved as {filename}")


# ==========================================
# 4. ENSEMBLE BNN & STOCHASTIC CHANNEL LAYER
# ==========================================
class StochasticChannelLayer(nn.Module):
    """
    Simulates the physical wireless channel.
    Samples new weights on every forward pass. 
    Uses a seeded generator during training to lock in a specific w^(l_0) sequence.
    """
    def __init__(self, K, hid_dim):
        super().__init__()
        self.K = K
        self.hid_dim = hid_dim
        # CRITICAL: We explicitly assign the sequence generator to the CPU. 
        # This guarantees that the sampled sequence is identical and repeatable 
        # across different hardware (MPS, CUDA, CPU), satisfying the fixed w^(l_0) condition.
        self.train_gen = torch.Generator(device='cpu')
        self.train_gen.manual_seed(1337)

    def forward(self, x, is_train=True):
        # x shape: (K, Batch, hid_dim)
        B = x.shape[1]
        
        if is_train:
            # Sample from P_tr sequence. Generate on CPU for hardware-agnostic reproducibility, then move to device.
            m = torch.randn(self.K, B, self.hid_dim, generator=self.train_gen, device='cpu').to(device) * STD_M_TR + MU_M_TR
            b = torch.randn(self.K, B, self.hid_dim, generator=self.train_gen, device='cpu').to(device) * STD_B_TR + MU_B_TR
        else:
            # Sample from P_te dynamically. Unseeded, representing novel real-world channel conditions.
            m = torch.randn(self.K, B, self.hid_dim, device=device) * STD_M_TE + MU_M_TE
            b = torch.randn(self.K, B, self.hid_dim, device=device) * STD_B_TE + MU_B_TE
            
        return x * m + b


class VectorizedBNNEnsemble(nn.Module):
    """
    Vectorized representation of K Bayesian Neural Networks.
    Models the joint distribution and the marginal Mixture distribution efficiently.
    """
    def __init__(self, K, in_dim, hid_dim, out_dim):
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        
        # Dimensions for Layer 1 and Layer 2 weights + biases
        self.D1 = in_dim * hid_dim + hid_dim
        self.D2 = hid_dim * out_dim + out_dim
        self.D = self.D1 + self.D2
        
        # Learnable baseline parameters: Mean and Rho (for log-variance)
        self.mu = nn.Parameter(torch.randn(K, self.D) * 0.1)
        self.rho = nn.Parameter(torch.randn(K, self.D) * 0.1 - 3.0)
        
        # Instantiate the stochastic channel layer
        self.channel_layer = StochasticChannelLayer(K, hid_dim)

    def get_sigma(self):
        # Softplus ensures strict positivity for standard deviation
        return torch.log1p(torch.exp(self.rho))

    def sample_theta(self, num_samples=1):
        """Reparameterization trick to sample learned weights \tilde{W}."""
        sigma = self.get_sigma()
        eps = torch.randn(num_samples, self.K, self.D, device=device)
        return self.mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

    def forward(self, x, theta, is_train=True):
        """
        Forward pass for all K models simultaneously over the batch.
        x: (Batch, In_dim)
        theta: (K, D) sampled learned weights
        """
        # Reshape flat theta into Layer 1 and Layer 2 matrices
        w1 = theta[:, :self.in_dim*self.hid_dim].view(self.K, self.in_dim, self.hid_dim)
        b1 = theta[:, self.in_dim*self.hid_dim:self.D1].view(self.K, self.hid_dim)
        
        w2 = theta[:, self.D1:self.D1+self.hid_dim*self.out_dim].view(self.K, self.hid_dim, self.out_dim)
        b2 = theta[:, self.D1+self.hid_dim*self.out_dim:].view(self.K, self.out_dim)
        
        # Broadcast batch across K models
        B = x.shape[0]
        x_k = x.unsqueeze(0).expand(self.K, B, self.in_dim)
        
        # --- Layer 1 ---
        h1 = torch.bmm(x_k, w1) + b1.unsqueeze(1)
        h1 = F.relu(h1)
        
        # --- Stochastic Wireless Channel Layer (l_0) ---
        h1_ch = self.channel_layer(h1, is_train=is_train)
        
        # --- Layer 2 ---
        out = torch.bmm(h1_ch, w2) + b2.unsqueeze(1)
        
        return out # Shape: (K, Batch, Out_dim)

# ==========================================
# 5. INFORMATION THEORETIC BOUND CALCULATIONS
# ==========================================
def log_gaussian(theta, mu, sigma):
    """Evaluates log N(theta | mu, sigma^2). Returns shape (Samples, K)"""
    var = sigma**2 + 1e-8
    log_scale = torch.log(var) * 0.5 + 0.5 * np.log(2 * np.pi)
    diff = (theta - mu.unsqueeze(0))**2 / (2 * var.unsqueeze(0))
    return -torch.sum(log_scale.unsqueeze(0) + diff, dim=-1)

def log_prior(theta, prior_lambda):
    """Evaluates log Q(theta) where Q is N(0, lambda*I)."""
    var = prior_lambda
    log_scale = 0.5 * np.log(var) + 0.5 * np.log(2 * np.pi)
    diff = theta**2 / (2 * var)
    return -torch.sum(log_scale + diff, dim=-1)

def compute_expected_kl(model):
    """Calculates E_w[ D(P_{W|w,S} || Q) ] using closed-form Gaussian KL."""
    mu = model.mu
    sigma = model.get_sigma()
    var = sigma**2
    # Analytical KL per ensemble member
    kl_k = 0.5 * torch.sum(var/PRIOR_LAMBDA + mu**2/PRIOR_LAMBDA - 1 - torch.log(var/PRIOR_LAMBDA), dim=-1)
    return kl_k.mean() # Expectation over training channels P_tr

def compute_mixture_kl_and_mi(model, num_samples=3):
    """
    Approximates the intractable terms via Monte Carlo sampling.
    Returns Standard Marginal KL and Mutual Information I(W; w | S).
    """
    theta = model.sample_theta(num_samples) # Shape: (S, K, D)
    mu = model.mu
    sigma = model.get_sigma()
    
    # 1. Log density of samples under their OWN posteriors (P_{W|w,S})
    log_q_k = log_gaussian(theta, mu, sigma) # Shape: (S, K)
    
    # 2. Log density of samples under the MIXTURE marginal (P_{W|S})
    # Expand dims to compute all pairwise densities: N(theta_{s,k} | mu_j, sigma_j)
    theta_exp = theta.unsqueeze(2)        # (S, K, 1, D)
    mu_exp = mu.unsqueeze(0).unsqueeze(0) # (1, 1, K_eval, D)
    sigma_exp = sigma.unsqueeze(0).unsqueeze(0) 
    
    var_exp = sigma_exp**2 + 1e-8
    log_scale = torch.log(var_exp) * 0.5 + 0.5 * np.log(2 * np.pi)
    diff = (theta_exp - mu_exp)**2 / (2 * var_exp)
    
    log_q_all = -torch.sum(log_scale + diff, dim=-1) # Shape: (S, K, K_eval)
    
    # Use logsumexp for numerical stability
    log_p_mix = torch.logsumexp(log_q_all, dim=2) - np.log(model.K) # Shape: (S, K)
    
    # Mutual Information: E[log P_{W|w,S} - log P_{W|S}]
    mi = (log_q_k - log_p_mix).mean()
    
    # Standard KL: E_{W~P_mix} [log P_{W|S} - log Q]
    log_q_prior = log_prior(theta, PRIOR_LAMBDA) # Shape: (S, K)
    kl_mix = (log_p_mix - log_q_prior).mean()
    
    return kl_mix, mi

# ==========================================
# 6. TRAINING LOOP & REGIMES
# ==========================================
def train_scenario(scenario_name, loader, n_samples):
    print(f"\n--- Training Scenario: {scenario_name.upper()} ---")
    model = VectorizedBNNEnsemble(K_ENSEMBLE, IN_DIM, HIDDEN_DIM, OUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    complexity_term = np.log(np.sqrt(n_samples) / EPSILON)
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_reg = 0.0
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Forward pass (1 MC sample for speed during CE calculation)
            theta = model.sample_theta(1).squeeze(0)
            # Pass is_train=True to sample from the seeded P_tr sequence
            out = model(batch_x, theta, is_train=True)
            
            # Cross Entropy Loss
            y_expanded = batch_y.unsqueeze(0).expand(model.K, -1)
            ce_loss = F.cross_entropy(out.reshape(-1, OUT_DIM), y_expanded.reshape(-1))
            
            loss = ce_loss
            reg_val = 0.0
            
            if scenario_name == 'b_standard_pac_bayes' or scenario_name == 'c_proposed_bound':
                # MC Sampling for bounds
                kl_mix, mi = compute_mixture_kl_and_mi(model, num_samples=3)
                
                if scenario_name == 'b_standard_pac_bayes':
                    # Standard Regularization Formula
                    reg = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (torch.clamp(kl_mix, min=0) + complexity_term))
                    loss = ce_loss + (REG_COEFF * reg)
                    reg_val = reg.item()
                    
                elif scenario_name == 'c_proposed_bound':
                    # Proposed Framework Formula
                    expected_kl = compute_expected_kl(model)
                    
                    term1 = torch.sqrt(2 * SIGMA_0_SQ * (torch.clamp(mi, min=0) + KL_CH_TOTAL))
                    term2 = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (expected_kl + complexity_term))
                    reg = term1 + term2
                    
                    loss = ce_loss + (REG_COEFF * reg)
                    reg_val = reg.item()

            loss.backward()
            optimizer.step()
            
            epoch_loss += ce_loss.item()
            epoch_reg += (REG_COEFF * reg_val) # Log the scaled applied bound
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | CE Loss: {epoch_loss/len(loader):.4f} | Scaled Reg Bound: {epoch_reg/len(loader):.4f}")
            
    return model

# ==========================================
# 7. EVALUATION ON INFERENCE CHANNEL (P_te)
# ==========================================
def evaluate_inference(model, loader):
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        # Evaluate using expected weights (mean of the posterior) to reduce noise.
        theta_mean = model.mu
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass through Inference Channel (samples dynamically from P_te)
            out = model(batch_x, theta_mean, is_train=False)
            
            # Prediction via majority voting across the ensemble
            probs = F.softmax(out, dim=-1)       # (K, Batch, 2)
            avg_probs = probs.mean(dim=0)        # (Batch, 2)
            preds = avg_probs.argmax(dim=-1)     # (Batch,)
            
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
    return correct / total

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    loader, n_samples = get_dataloaders()
    
    # Scenario A: Standard ERM
    model_erm = train_scenario('a_erm', loader, n_samples)
    acc_erm = evaluate_inference(model_erm, loader)
    
    # Scenario B: Standard PAC-Bayes Regularization
    model_pac = train_scenario('b_standard_pac_bayes', loader, n_samples)
    acc_pac = evaluate_inference(model_pac, loader)
    
    # Scenario C: Proposed Bound Regularization
    model_prop = train_scenario('c_proposed_bound', loader, n_samples)
    acc_prop = evaluate_inference(model_prop, loader)
    
    print("\n" + "="*50)
    print("FINAL INFERENCE RESULTS (EVALUATED ON P_te)")
    print("="*50)
    print(f"Standard ERM Accuracy:                {acc_erm*100:.2f}%")
    print(f"Standard PAC-Bayes Accuracy:          {acc_pac*100:.2f}%")
    print(f"Proposed Bound Reg Accuracy:          {acc_prop*100:.2f}%")
    print("="*50)