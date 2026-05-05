import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================
# 0. OUTPUT PATHS
# ==========================================
RESULTS_DIR = os.path.join('results', 'test6')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
WEIGHTS_DIR = os.path.join(RESULTS_DIR, 'weights')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ==========================================
# 1. HARDWARE & PATH CONFIGURATION (CRITICAL)
# ==========================================
# Exact device string requested for Mac M-series fallback
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 2. HYPERPARAMETERS & CHANNEL DISTRIBUTIONS
# ==========================================
N_SAMPLES = 1000       # Total samples in the dataset
K_ENSEMBLE = 10        # Number of BNNs in the ensemble
HIDDEN_DIM = 16       # INCREASED: Give ERM enough capacity to overfit the channel
IN_DIM = 2
OUT_DIM = 2
BATCH_SIZE = 64        # Reduced batch size for noisier gradients
EPOCHS = 150           # Increased epochs to ensure ERM fully memorizes
PRIOR_LAMBDA = 1.0     # Variance of isotropic Gaussian prior
EPSILON = 0.05         # PAC-Bayes confidence parameter
SIGMA_SQ = 1.0         # Assumed sub-Gaussian parameter
SIGMA_0_SQ = 1.0       # Assumed sub-Gaussian parameter
REG_COEFF = 0.05       # Adjusted: Give the regularizer a slightly stronger voice
REG_ALPHA = 0.5        # Weight for MI term in the proposed bound
REG_BETA = 0.5         # Weight for KL(P_{W|w,S} || Q) term in the proposed bound

# MI estimation hyperparameters (one-step channel-conditioned approximation)
MI_NUM_CHANNEL_SAMPLES = 5
MI_NUM_WEIGHT_SAMPLES = 1
MI_STEP_LR = 1e-2
MI_CALC_EVERY = 1

# Channel Distributions (Train) - DECEPTIVELY CLEAN
# ERM will become overconfident and build fragile decision boundaries.
# MU_M_TR, STD_M_TR = 1.0, 0.01  # Almost no fading variation
# MU_B_TR, STD_B_TR = 0.0, 0.01  # Almost no AWGN
# MU_M_TR, STD_M_TR = 1.0, 0.1  # Almost no fading variation
# MU_B_TR, STD_B_TR = 0.0, 0.1  # Almost no AWGN
MU_M_TR, STD_M_TR = 0.5, 0.5  # Almost no fading variation
MU_B_TR, STD_B_TR = 0.0, 1.0  # Almost no AWGN

# Channel Distributions (Inference/Test) - HARSH REALITY
# ERM's fragile boundaries will fail here. Bayesian smoothing should win.
MU_M_TE, STD_M_TE = 0.5, 0.5   # Severe fading
MU_B_TE, STD_B_TE = 0.0, 1.0   # Heavy noise

def kl_gaussian_1d(mu1, std1, mu2, std2):
    """Closed form KL D(N_1 || N_2)"""
    return np.log(std2/std1) + (std1**2 + (mu1 - mu2)**2) / (2 * std2**2) - 0.5

# Closed form KL between Train and Test channels
kl_ch_m = HIDDEN_DIM * kl_gaussian_1d(MU_M_TR, STD_M_TR, MU_M_TE, STD_M_TE)
kl_ch_b = HIDDEN_DIM * kl_gaussian_1d(MU_B_TR, STD_B_TR, MU_B_TE, STD_B_TE)
KL_CH_TOTAL = kl_ch_m + kl_ch_b
print(f"Constant D(P_art || P_ch) = {KL_CH_TOTAL:.4f}")

def estimate_expected_channel_norm(hidden_dim, mu_m_te, mu_b_te, std_m_te, std_b_te, norm_type='frobenius', mc_samples=1000, device='cpu'):
    """
    Monte Carlo estimation of E[ || W' - (I, 0) || ]
    Where W' = (M, B). Therefore W' - (I, 0) = (M - I, B).
    """
    # M_diff represents M - I ~ N(0, sigma)
    M_diff = torch.randn(mc_samples, hidden_dim, hidden_dim, device=device) * std_m_te + mu_m_te - 1.0
    # B represents B - 0 ~ N(0, sigma)
    B = torch.randn(mc_samples, hidden_dim, 1, device=device) * std_b_te + mu_b_te
    
    # Concatenate along the last dimension to represent the augmented matrix (M-I, B)
    W_diff = torch.cat([M_diff, B], dim=2)
    
    if norm_type == 'frobenius':
        norms = torch.linalg.matrix_norm(W_diff, ord='fro')
    elif norm_type == 'spectral':
        norms = torch.linalg.matrix_norm(W_diff, ord=2)
    else:
        raise ValueError("norm_type must be 'frobenius' or 'spectral'")
        
    return norms.mean()

# channel penalty
CH_PENALTY = estimate_expected_channel_norm(HIDDEN_DIM, MU_M_TE, MU_B_TE, STD_M_TE, STD_B_TE, device=device)


# ==========================================
# 3. DATA GENERATION
# ==========================================
# Personalized dataset path handling
data_path = os.environ.get('DATASET', './data')
os.makedirs(data_path, exist_ok=True)

def get_dataloaders(n_samples, batch_size=32):
    """Generates make_moons dataset and saves to DATASET path."""
    dataset_file = os.path.join(data_path, 'moons_data.pt')
    
    if not os.path.exists(dataset_file):
        print(f"Generating data and saving to {dataset_file}...")
        X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=42)

        visualize_and_save_dataset(X, y, filename=os.path.join(data_path, "moons_viz.png"), title="Original Moons Dataset")

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        torch.save((X_tensor, y_tensor), dataset_file)
    else:
        print(f"Loading existing data from {dataset_file}...")
        X_tensor, y_tensor = torch.load(dataset_file)
    
        
    # 80/20 Train-Test Split
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_size

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

    def forward(self, x, mode='perfect', channel_override=None):
        # x shape: (K, Batch, hid_dim)
        B = x.shape[1]
        
        if mode == 'given':
            if channel_override is None:
                raise ValueError("channel_override must be provided when mode='given'.")
            m, b = channel_override
        elif mode == 'perfect':
            # No channel effect, perfect transmission
            m = torch.ones_like(x, device=x.device)
            b = torch.zeros_like(x, device=x.device)
        elif mode == 'train':
            # Sample from P_art sequence. Generate on CPU for hardware-agnostic reproducibility, then move to device.
            m = torch.randn(self.K, B, self.hid_dim, generator=self.train_gen, device='cpu').to(x.device) * STD_M_TR + MU_M_TR
            b = torch.randn(self.K, B, self.hid_dim, generator=self.train_gen, device='cpu').to(x.device) * STD_B_TR + MU_B_TR
        elif mode == 'test':
            # Sample from P_ch dynamically. Unseeded, representing novel real-world channel conditions.
            m = torch.randn(self.K, B, self.hid_dim, device=x.device) * STD_M_TE + MU_M_TE
            b = torch.randn(self.K, B, self.hid_dim, device=x.device) * STD_B_TE + MU_B_TE
        else:
            raise ValueError(f"Invalid mode '{mode}' for StochasticChannelLayer. Choose from 'perfect', 'train', or 'test'.")
            
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

    def forward(self, x, theta, mode='perfect', channel_override=None):
        """
        Forward pass for all K models simultaneously over the batch.
        x: (Batch, In_dim)
        theta: (K, D) sampled learned weights
        mode: 'perfect', 'train', or 'test' to control channel behavior
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
        h1_ch = self.channel_layer(h1, mode=mode, channel_override=channel_override)
        
        # --- Layer 2 ---
        out = torch.bmm(h1_ch, w2) + b2.unsqueeze(1)
        
        return out # Shape: (K, Batch, Out_dim)
    
    def get_sampled_weights(self, theta):
        """
        Returns a list of sampled parameter tensors (views) from a flat theta.
        theta: (K, D)
        """
        w1 = theta[:, :self.in_dim*self.hid_dim].view(self.K, self.in_dim, self.hid_dim)
        b1 = theta[:, self.in_dim*self.hid_dim:self.D1].view(self.K, self.hid_dim)
        w2 = theta[:, self.D1:self.D1+self.hid_dim*self.out_dim].view(self.K, self.hid_dim, self.out_dim)
        b2 = theta[:, self.D1+self.hid_dim*self.out_dim:].view(self.K, self.out_dim)
        return [w1, b1, w2, b2]

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
    return kl_k.mean() # Expectation over training channels P_art

def compute_mixture_kl(model, num_samples=3):
    """
    Approximates the intractable mixture KL via Monte Carlo sampling.
    Returns Standard Marginal KL.
    """
    theta = model.sample_theta(num_samples) # Shape: (S, K, D)
    mu = model.mu
    sigma = model.get_sigma()
    
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
    
    # Standard KL: E_{W~P_mix} [log P_{W|S} - log Q]
    log_q_prior = log_prior(theta, PRIOR_LAMBDA) # Shape: (S, K)
    kl_mix = (log_p_mix - log_q_prior).mean()
    
    return kl_mix

def _gaussian_log_prob_diag(x, mu, sigma):
    """Log prob of diagonal Gaussian. Shapes: x, mu, sigma -> (..., D). Returns (...)."""
    var = sigma**2 + 1e-8
    log_scale = 0.5 * torch.log(2 * torch.pi * var)
    diff = (x - mu)**2 / (2 * var)
    return -(log_scale + diff).sum(dim=-1)

def estimate_channel_mi_one_step(model, batch_x, batch_y, num_channel_samples=5, num_weight_samples=1, step_lr=1e-2):
    """
    Monte Carlo estimate of I_S(\tilde{W}; W^{(l_0)}) using one-step channel-conditioned updates.
    This simulates the dependency introduced by the last-step channel sample.
    """
    model.train()
    B = batch_x.shape[0]

    mu_base = model.mu
    rho_base = model.rho

    mu_list = []
    sigma_list = []

    for _ in range(num_channel_samples):
        # Sample a single channel realization from P_art (training channel)
        m = torch.randn(model.K, B, model.hid_dim, device=batch_x.device) * STD_M_TR + MU_M_TR
        b = torch.randn(model.K, B, model.hid_dim, device=batch_x.device) * STD_B_TR + MU_B_TR

        theta = model.sample_theta(1).squeeze(0)
        out = model(batch_x, theta, mode='given', channel_override=(m, b))
        y_expanded = batch_y.unsqueeze(0).expand(model.K, -1)
        ce_loss = F.cross_entropy(out.reshape(-1, OUT_DIM), y_expanded.reshape(-1))

        grads = torch.autograd.grad(ce_loss, [mu_base, rho_base], retain_graph=False, create_graph=False)
        grad_mu, grad_rho = grads

        mu_k = (mu_base - step_lr * grad_mu).detach()
        rho_k = (rho_base - step_lr * grad_rho).detach()
        sigma_k = torch.log1p(torch.exp(rho_k))

        mu_list.append(mu_k)
        sigma_list.append(sigma_k)

    mu_stack = torch.stack(mu_list, dim=0)       # (C, K, D)
    sigma_stack = torch.stack(sigma_list, dim=0) # (C, K, D)

    # Sample weights from each conditional posterior
    eps = torch.randn(num_weight_samples, *mu_stack.shape, device=batch_x.device)
    w_samples = mu_stack.unsqueeze(0) + sigma_stack.unsqueeze(0) * eps  # (S, C, K, D)

    # Log prob under corresponding conditional
    log_q_cond = _gaussian_log_prob_diag(w_samples, mu_stack.unsqueeze(0), sigma_stack.unsqueeze(0))  # (S, C, K)

    # Log prob under the marginal mixture over channels
    # Evaluate each sample against all channel-conditioned Gaussians
    w_exp = w_samples.unsqueeze(2)                     # (S, C, 1, K, D)
    mu_exp = mu_stack.unsqueeze(0).unsqueeze(1)        # (1, 1, C, K, D)
    sigma_exp = sigma_stack.unsqueeze(0).unsqueeze(1)  # (1, 1, C, K, D)

    log_q_all = _gaussian_log_prob_diag(w_exp, mu_exp, sigma_exp)  # (S, C, C, K)
    log_q_mix = torch.logsumexp(log_q_all, dim=2) - np.log(num_channel_samples)  # (S, C, K)

    mi = (log_q_cond - log_q_mix).mean()
    return mi




def evaluate_model(model, loader, mode='test'):
    """
    Returns (avg_loss, accuracy) using theta mean and the specified channel mode.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        theta_mean = model.mu
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = model(batch_x, theta_mean, mode=mode)

            avg_logits = out.mean(dim=0)
            loss = F.cross_entropy(avg_logits, batch_y)

            probs = F.softmax(out, dim=-1)
            avg_probs = probs.mean(dim=0)
            preds = avg_probs.argmax(dim=-1)

            total_loss += loss.item() * batch_y.size(0)
            total_correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def plot_training_metrics(history, scenario_name, mode):
    """Plots and saves loss/accuracy curves."""
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    title = f"{scenario_name.upper()} ({mode})"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = os.path.join(PLOTS_DIR, f"metrics_{scenario_name}_{mode}.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_bound_decomposition(history, scenario_name, mode):
    """Plots and saves bound decomposition curves if available."""
    if not history.get('bound_total'):
        return
    if max(history['bound_total']) == 0.0 and max(history['bound_term1']) == 0.0 and max(history['bound_term2']) == 0.0:
        return

    epochs = list(range(1, len(history['bound_total']) + 1))
    series = [
        ('Bound Total', history['bound_total']),
        ('Bound Term 1', history['bound_term1']),
        ('Bound Term 2', history['bound_term2']),
        ('MI', history.get('mi', [])),
        ('KL_CH_TOTAL', history.get('kl_ch_total', [])),
        ('E[KL]', history.get('expected_kl', [])),
        ('K_hat', history.get('k_hat', [])),
        ('Channel Penalty', history.get('channel_penalty', [])),
        ('K_hat * Channel Penalty', history.get('k_hat_channel_penalty', []))
    ]

    plotted = [(label, values) for label, values in series if values and max(values) != 0.0]
    if not plotted:
        return

    n_rows = len(plotted)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, (label, values) in zip(axes, plotted):
        ax.plot(epochs, values, label=label)
        ax.set_ylabel(label)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Epoch')

    title = f"Bound Decomposition: {scenario_name.upper()} ({mode})"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = os.path.join(PLOTS_DIR, f"bound_{scenario_name}_{mode}.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 6. TRAINING LOOP & REGIMES
# ==========================================
def train_scenario(scenario_name, loader, n_samples, mode='perfect', objective='bound'):
    print(f"\n--- Training Scenario: {scenario_name.upper()} ---")
    # if scenario_name == 'erm' or mode == 'perfect':
    #     model = VectorizedBNNEnsemble(1, IN_DIM, HIDDEN_DIM, OUT_DIM).to(device)
    # else:
    model = VectorizedBNNEnsemble(K_ENSEMBLE, IN_DIM, HIDDEN_DIM, OUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    complexity_term = np.log(np.sqrt(n_samples) / EPSILON)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'bound_total': [],
        'bound_term1': [],
        'bound_term2': [],
        'mi': [],
        'kl_ch_total': [],
        'expected_kl': [],
        'k_hat': [],
        'channel_penalty': [],
        'k_hat_channel_penalty': []
    }

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_reg = 0.0
        epoch_bound_total = 0.0
        epoch_term1 = 0.0
        epoch_term2 = 0.0
        epoch_mi = 0.0
        mi_count = 0
        epoch_kl_ch_total = 0.0
        epoch_expected_kl = 0.0
        epoch_k_hat = 0.0
        epoch_channel_penalty = 0.0
        epoch_k_hat_channel_penalty = 0.0

        model.train()
        
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Forward pass (1 MC sample for speed during CE calculation)
            theta = model.sample_theta(1).squeeze(0)
            # Pass mode='train' to sample from the seeded P_art sequence
            out = model(batch_x, theta, mode=mode)
            
            # Cross Entropy Loss
            y_expanded = batch_y.unsqueeze(0).expand(model.K, -1)
            ce_loss = F.cross_entropy(out.reshape(-1, OUT_DIM), y_expanded.reshape(-1))
            
            loss = ce_loss
            reg_val = 0.0
            term1_val = 0.0
            term2_val = 0.0
            mi_val = 0.0
            kl_ch_total_val = 0.0
            expected_kl_val = 0.0
            k_hat_val = 0.0
            channel_penalty_val = 0.0
            k_hat_channel_penalty_val = 0.0
            
            if scenario_name == 'pac_bayes' or scenario_name == 'proposed':
                # MC Sampling for bounds
                kl_mix = compute_mixture_kl(model, num_samples=3)
                
                if scenario_name == 'pac_bayes':
                    # Standard Regularization Formula
                    reg = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (torch.clamp(kl_mix, min=0) + complexity_term))
                    loss = ce_loss + (REG_COEFF * reg)
                    reg_val = reg.item()
                    term1_val = reg.item()
                    
                elif scenario_name == 'proposed':
                    # Proposed Framework Formula
                    expected_kl = compute_expected_kl(model)
                    expected_kl_val = expected_kl.item()
                    if mode == 'perfect':
                        # Lipschitz
                        grad_theta = torch.autograd.grad(
                            ce_loss,
                            theta,
                            create_graph=True,
                            retain_graph=True
                        )[0]
                        K_hat = torch.norm(grad_theta, p=2)
                        k_hat_val = K_hat.item()

                        channel_penalty_val = CH_PENALTY.item()

                        term1 = K_hat * CH_PENALTY
                        term2 = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (expected_kl + complexity_term))
                        reg = term1 + term2
                        term1_val = term1.item()
                        term2_val = term2.item()
                        k_hat_channel_penalty_val = term1_val
                        reg_val = reg.item()

                        if objective == 'bound':
                            loss = ce_loss + (REG_COEFF * reg)
                        elif objective == 'heuristic':
                            loss = ce_loss + (REG_COEFF * (REG_ALPHA * K_hat * CH_PENALTY + REG_BETA * expected_kl))
                    elif mode == 'train':
                        # Channel-conditioned MI estimate (computed once per epoch by default)
                        if (batch_idx == 0) and ((epoch + 1) % MI_CALC_EVERY == 0):
                            mi = estimate_channel_mi_one_step(
                                model,
                                batch_x,
                                batch_y,
                                num_channel_samples=MI_NUM_CHANNEL_SAMPLES,
                                num_weight_samples=MI_NUM_WEIGHT_SAMPLES,
                                step_lr=MI_STEP_LR,
                            )
                            mi_val = mi.item()
                            mi_count += 1
                        else:
                            mi = torch.tensor(0.0, device=device)

                        term1 = torch.sqrt(2 * SIGMA_0_SQ * (torch.clamp(mi, min=0) + KL_CH_TOTAL))
                        term2 = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (expected_kl + complexity_term))
                        reg = term1 + term2
                        term1_val = term1.item()
                        term2_val = term2.item()
                        if mi_val == 0.0 and mi_count == 0:
                            mi_val = 0.0
                        kl_ch_total_val = KL_CH_TOTAL
                        
                        if objective == 'bound':
                            loss = ce_loss + (REG_COEFF * reg)
                        elif objective == 'heuristic':
                            loss = ce_loss + (REG_COEFF * (REG_ALPHA * mi + REG_BETA * expected_kl))
                        reg_val = reg.item()

            loss.backward()
            optimizer.step()
            
            epoch_loss += ce_loss.item()
            epoch_reg += reg_val # Log the scaled applied bound
            epoch_bound_total += reg_val
            epoch_term1 += term1_val
            epoch_term2 += term2_val
            epoch_mi += mi_val
            epoch_kl_ch_total += kl_ch_total_val
            epoch_expected_kl += expected_kl_val
            epoch_k_hat += k_hat_val
            epoch_channel_penalty += channel_penalty_val
            epoch_k_hat_channel_penalty += k_hat_channel_penalty_val

        train_loss, train_acc = evaluate_model(model, loader, mode=mode)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['bound_total'].append(epoch_bound_total / len(loader))
        history['bound_term1'].append(epoch_term1 / len(loader))
        history['bound_term2'].append(epoch_term2 / len(loader))
        history['mi'].append(epoch_mi / max(mi_count, 1))
        history['kl_ch_total'].append(epoch_kl_ch_total / len(loader))
        history['expected_kl'].append(epoch_expected_kl / len(loader))
        history['k_hat'].append(epoch_k_hat / len(loader))
        history['channel_penalty'].append(epoch_channel_penalty / len(loader))
        history['k_hat_channel_penalty'].append(epoch_k_hat_channel_penalty / len(loader))

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"CE Loss: {epoch_loss/len(loader):.4f} | "
                f"Reg Bound: {epoch_reg/len(loader):.4f} | "
                f"Bound T1: {epoch_term1/len(loader):.4f} | "
                f"Bound T2: {epoch_term2/len(loader):.4f} | "
                f"Train Acc: {train_acc*100:.2f}%"
            )

    weights_path = os.path.join(WEIGHTS_DIR, f"weights_{scenario_name}_{mode}.pth")
    torch.save(model.state_dict(), weights_path)
    plot_training_metrics(history, scenario_name, mode)
    plot_bound_decomposition(history, scenario_name, mode)
            
    return model

# ==========================================
# 7. EVALUATION ON INFERENCE CHANNEL (P_ch)
# ==========================================
def evaluate_inference(model, loader):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        # Evaluate using expected weights (mean of the posterior) to reduce noise.
        theta_mean = model.mu

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass through Inference Channel (samples dynamically from P_ch)
            out = model(batch_x, theta_mean, mode='test')

            avg_logits = out.mean(dim=0)
            loss = F.cross_entropy(avg_logits, batch_y)

            # Prediction via majority voting across the ensemble
            probs = F.softmax(out, dim=-1)       # (K, Batch, 2)
            avg_probs = probs.mean(dim=0)        # (Batch, 2)
            preds = avg_probs.argmax(dim=-1)     # (Batch,)

            total_loss += loss.item() * batch_y.size(0)
            total_correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    train_loader, test_loader, n_trains = get_dataloaders(N_SAMPLES, batch_size=BATCH_SIZE)
    
    # Scenario A: Standard ERM + perfect channel
    model_erm_perfect = train_scenario('erm', train_loader, n_trains, mode='perfect', objective='bound')
    loss_erm_perfect, acc_erm_perfect = evaluate_inference(model_erm_perfect, test_loader)

    # Scenario B: Standard ERM + train channel (overfitting to P_art)
    model_erm = train_scenario('erm', train_loader, n_trains, mode='train', objective='bound')
    loss_erm, acc_erm = evaluate_inference(model_erm, test_loader)
    
    # Scenario C: Proposed Bound Regularization
    model_prop = train_scenario('proposed', train_loader, n_trains, mode='train', objective='bound')
    loss_prop, acc_prop = evaluate_inference(model_prop, test_loader)

    # Scenario C2: Proposed Bound Regularization with heuristic objective (no direct bound optimization)
    model_prop_heuristic = train_scenario('proposed', train_loader, n_trains, mode='train', objective='heuristic')
    loss_prop_heuristic, acc_prop_heuristic = evaluate_inference(model_prop_heuristic, test_loader)

    # Scenario D: Proposed Bound + perfect channel
    model_prop_perfect = train_scenario('proposed', train_loader, n_trains, mode='perfect', objective='bound')
    loss_prop_perfect, acc_prop_perfect = evaluate_inference(model_prop_perfect, test_loader)

    # Scenario D2: Proposed Bound + perfect channel with heuristic objective
    model_prop_perfect_heuristic = train_scenario('proposed', train_loader, n_trains, mode='perfect', objective='heuristic')
    loss_prop_perfect_heuristic, acc_prop_perfect_heuristic = evaluate_inference(model_prop_perfect_heuristic, test_loader)
    
    print("\n" + "="*50)
    print("FINAL INFERENCE RESULTS (EVALUATED ON P_ch)")
    print("="*50)
    print(f"Standard ERM + Perfect Channel Loss/Acc: {loss_erm_perfect:.4f} / {acc_erm_perfect*100:.2f}%")
    print(f"Standard ERM Loss/Acc:                {loss_erm:.4f} / {acc_erm*100:.2f}%")
    # print(f"Standard PAC-Bayes Loss/Acc:          {loss_pac:.4f} / {acc_pac*100:.2f}%")
    print(f"Proposed Bound Reg Loss/Acc:          {loss_prop:.4f} / {acc_prop*100:.2f}%")
    print(f"Proposed Heuristic Reg Loss/Acc:      {loss_prop_heuristic:.4f} / {acc_prop_heuristic*100:.2f}%")
    print(f"Proposed Bound + Perfect Channel Loss/Acc: {loss_prop_perfect:.4f} / {acc_prop_perfect*100:.2f}%")
    print(f"Proposed Bound + Perfect Channel (Heuristic) Loss/Acc: {loss_prop_perfect_heuristic:.4f} / {acc_prop_perfect_heuristic*100:.2f}%")
    print("="*50)