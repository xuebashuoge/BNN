import os
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ERM using deterministic, test for different train conditions

# ==========================================
# 0. OUTPUT PATHS
# ==========================================
RESULTS_DIR = os.path.join('results', 'test9')
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
HIDDEN_DIM = 64       # INCREASED: Give ERM enough capacity to overfit the channel
IN_DIM = 2
OUT_DIM = 2
MOONS_NOISE = 0.3      # make_moons noise level
BATCH_SIZE = 64        # Reduced batch size for noisier gradients
EPOCHS = 150           # Increased epochs to ensure ERM fully memorizes
LR_BASE = 1e-2         # Adjusted base learning rate
LR_DECAY_STEP = 0     # StepLR decay period (epochs)
LR_DECAY_GAMMA = 0.5   # StepLR decay factor
PRIOR_LAMBDA = 1.0     # Variance of isotropic Gaussian prior
EPSILON = 0.025         # PAC-Bayes confidence parameter
SIGMA_SQ = 1.0         # Assumed sub-Gaussian parameter
SIGMA_0_SQ = 1.0       # Assumed sub-Gaussian parameter
REG_COEFF = 0.06       # Adjusted: Give the regularizer a slightly stronger voice
REG_ALPHA = 0.06        # Weight for MI term in the proposed bound
REG_BETA = 0.06         # Weight for KL(P_{W|w,S} || Q) term in the proposed bound
MI_MC_SAMPLES = 100       # MC samples for MI/KL mixture estimation (proposed)
SEED = 2
LIPSCHITZ_METHOD_PERFECT = "grad"  # "grad" or "analytical"

# Channel Distributions (Inference/Test) - HARSH REALITY
# ERM's fragile boundaries will fail here. Bayesian smoothing should win.
MU_M_TE, STD_M_TE = 0.5, 1.0
MU_B_TE, STD_B_TE = 0.0, 1.0   

# Channel Distributions (Train) - DECEPTIVELY CLEAN
# ERM will become overconfident and build fragile decision boundaries.
# MU_M_TR, STD_M_TR = 1.0, 0.01  # Almost no fading variation
# MU_B_TR, STD_B_TR = 0.0, 0.01  # Almost no AWGN
# MU_M_TR, STD_M_TR = 1.0, 0.1  # Almost no fading variation
# MU_B_TR, STD_B_TR = 0.0, 0.1  # Almost no AWGN
MU_M_TR, STD_M_TR = MU_M_TE, STD_M_TE  # Almost no fading variation
MU_B_TR, STD_B_TR = MU_B_TE, STD_B_TE  # Almost no AWGN

def kl_gaussian_1d(mu1, std1, mu2, std2):
    """Closed form KL D(N_1 || N_2)"""
    return np.log(std2/std1) + (std1**2 + (mu1 - mu2)**2) / (2 * std2**2) - 0.5

def compute_train_test_channel_kl(mu_m_tr, std_m_tr, mu_b_tr, std_b_tr):
    """Closed form KL divergence D(P_art || P_ch) for channel parameters."""
    kl_ch_m = HIDDEN_DIM * kl_gaussian_1d(mu_m_tr, std_m_tr, MU_M_TE, STD_M_TE)
    kl_ch_b = HIDDEN_DIM * kl_gaussian_1d(mu_b_tr, std_b_tr, MU_B_TE, STD_B_TE)
    return kl_ch_m + kl_ch_b

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
    
    is_mps = isinstance(device, torch.device) and device.type == "mps" or device == "mps"
    if norm_type == 'frobenius':
        norms = torch.linalg.matrix_norm(W_diff, ord='fro')
    elif norm_type == 'spectral':
        # MPS backend does not support SVD-based spectral norm yet; compute on CPU when needed.
        if is_mps:
            norms = torch.linalg.matrix_norm(W_diff.cpu(), ord=2).to(device)
        else:
            norms = torch.linalg.matrix_norm(W_diff, ord=2)
    else:
        raise ValueError("norm_type must be 'frobenius' or 'spectral'")
        
    return norms.mean()

# channel penalty
CH_PENALTY = estimate_expected_channel_norm(HIDDEN_DIM, MU_M_TE, MU_B_TE, STD_M_TE, STD_B_TE, norm_type='frobenius', device=device)
KL_CH_TOTAL = compute_train_test_channel_kl(MU_M_TR, STD_M_TR, MU_B_TR, STD_B_TR)
print(f"Constant D(P_art || P_ch) = {KL_CH_TOTAL:.4f}")


def set_train_channel_params(mu_m_tr, std_m_tr, mu_b_tr, std_b_tr):
    """
    Update global training channel parameters and dependent KL quantity.
    """
    global MU_M_TR, STD_M_TR, MU_B_TR, STD_B_TR, KL_CH_TOTAL
    MU_M_TR, STD_M_TR = float(mu_m_tr), float(std_m_tr)
    MU_B_TR, STD_B_TR = float(mu_b_tr), float(std_b_tr)
    KL_CH_TOTAL = compute_train_test_channel_kl(MU_M_TR, STD_M_TR, MU_B_TR, STD_B_TR)
    print(
        "Updated train channel:"
        f" MU_M_TR={MU_M_TR:.4f}, STD_M_TR={STD_M_TR:.4f},"
        f" MU_B_TR={MU_B_TR:.4f}, STD_B_TR={STD_B_TR:.4f},"
        f" KL_CH_TOTAL={KL_CH_TOTAL:.4f}"
    )


# ==========================================
# 3. DATA GENERATION
# ==========================================
# Personalized dataset path handling
data_path = os.environ.get('DATASET', './data')
os.makedirs(data_path, exist_ok=True)

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(n_samples, batch_size=32, seed=0, noise=0.3):
    """Generates make_moons dataset and saves to DATASET path."""
    noise_tag = f"{noise:.4f}".replace(".", "p")
    dataset_file = os.path.join(data_path, f"moons_data_n{n_samples}_noise_{noise_tag}.pt")
    
    if not os.path.exists(dataset_file):
        print(f"Generating data and saving to {dataset_file}...")
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

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
    generator = torch.Generator().manual_seed(seed)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
    
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

    def forward(self, x, mode='perfect'):
        # x shape: (K, Batch, hid_dim)
        B = x.shape[1]
        
        if mode == 'perfect':
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

        self.last_m = m
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
        self.is_bnn = True
        
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

    def forward(self, x, theta, mode='perfect'):
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
        h1_ch = self.channel_layer(h1, mode=mode)
        
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
    
    def compute_analytical_lipschitz(self, x, theta, mode='perfect'):
        """
        Calculates an analytical upper bound of the Lipschitz constant (gradient norm w.r.t weights)
        using spectral norms and max activations over the batch.
        """
        w1, b1, w2, b2 = self.get_sampled_weights(theta)
        
        # 1. Spectral Norm of Weight matrices
        # w2 shape is (K, hid_dim, out_dim). matrix_norm works over last two dims.
        # MPS fallback: SVD is not implemented on MPS for spectral norm.
        if w2.is_mps:
            w2_sn = torch.linalg.matrix_norm(w2.cpu(), ord=2).to(w2.device)
        else:
            w2_sn = torch.linalg.matrix_norm(w2, ord=2) # Shape: (K,)
        
        # 2. Forward pass to collect maximum activation norms
        with torch.no_grad():
            B = x.shape[0]
            x_k = x.unsqueeze(0).expand(self.K, B, self.in_dim)
            h1 = torch.bmm(x_k, w1) + b1.unsqueeze(1)
            h1 = F.relu(h1)
            h1_ch = self.channel_layer(h1, mode=mode)
            m = self.channel_layer.last_m
        
        # 3. Maximum bounds
        x_norm = torch.norm(x, p=2, dim=1).max() # scalar max over batch
        h1_ch_norm = torch.norm(h1_ch, p=2, dim=2).max(dim=1)[0] # Shape: (K,)
        m_max = torch.abs(m).max(dim=2)[0].max(dim=1)[0] if m is not None else torch.ones(self.K, device=device)
        
        # CE Lipschitz w.r.t logits is bounded by sqrt(2) approx 1.414
        ce_lip = 1.414 
        
        # Combine bounds: ||grad_theta|| <= sqrt(||grad_w2||^2 + ||grad_b2||^2 + ||grad_w1||^2 + ||grad_b1||^2)
        grad_w2_bound = h1_ch_norm
        grad_b2_bound = 1.0
        grad_w1_bound = x_norm * w2_sn * m_max
        grad_b1_bound = w2_sn * m_max
        
        k_hat_analytical = ce_lip * torch.sqrt(
            grad_w2_bound**2 + 
            grad_b2_bound**2 + 
            grad_w1_bound**2 + 
            grad_b1_bound**2
        )
        return k_hat_analytical.mean()


class DeterministicFC(nn.Module):
    """
    Standard 2-layer FC network with the same stochastic channel layer.
    Used as the baseline ERM model (non-Bayesian).
    """
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.channel_layer = StochasticChannelLayer(1, hid_dim)
        self.is_bnn = False

    def forward(self, x, mode='perfect'):
        h1 = F.relu(self.fc1(x))
        h1_ch = self.channel_layer(h1.unsqueeze(0), mode=mode).squeeze(0)
        return self.fc2(h1_ch)

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




def evaluate_model(model, loader, mode='test'):
    """
    Returns (avg_loss, accuracy) using theta mean and the specified channel mode.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        theta_mean = model.mu if getattr(model, "is_bnn", False) else None
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if getattr(model, "is_bnn", False):
                out = model(batch_x, theta_mean, mode=mode)
                avg_logits = out.mean(dim=0)
                loss = F.cross_entropy(avg_logits, batch_y)

                probs = F.softmax(out, dim=-1)
                avg_probs = probs.mean(dim=0)
                preds = avg_probs.argmax(dim=-1)
            else:
                logits = model(batch_x, mode=mode)
                loss = F.cross_entropy(logits, batch_y)
                preds = logits.argmax(dim=-1)

            total_loss += loss.item() * batch_y.size(0)
            total_correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def plot_training_metrics(history, scenario_name, mode, plot_suffix=None, title_extra=None):
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
    if title_extra:
        title = f"{title}\n{title_extra}"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    suffix = f"_{plot_suffix}" if plot_suffix else ""
    filename = os.path.join(PLOTS_DIR, f"metrics_{scenario_name}_{mode}{suffix}.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_bound_decomposition(history, scenario_name, mode, plot_suffix=None, title_extra=None):
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
    if title_extra:
        title = f"{title}\n{title_extra}"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    suffix = f"_{plot_suffix}" if plot_suffix else ""
    filename = os.path.join(PLOTS_DIR, f"bound_{scenario_name}_{mode}{suffix}.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def format_float_for_label(value):
    """Compact float string that is safe for filenames."""
    return f"{float(value):.4g}".replace("-", "neg").replace(".", "p")


def build_channel_plot_label(cfg, seed):
    return (
        f"seed{seed}_{cfg['axis']}_{format_float_for_label(cfg['axis_value'])}"
        f"_mm{format_float_for_label(cfg['mu_m_tr'])}"
        f"_sm{format_float_for_label(cfg['std_m_tr'])}"
        f"_mb{format_float_for_label(cfg['mu_b_tr'])}"
        f"_sb{format_float_for_label(cfg['std_b_tr'])}"
    )


def build_channel_title(cfg):
    return (
        "Train channel: "
        f"MU_M_TR={cfg['mu_m_tr']:.4g}, STD_M_TR={cfg['std_m_tr']:.4g}, "
        f"MU_B_TR={cfg['mu_b_tr']:.4g}, STD_B_TR={cfg['std_b_tr']:.4g}"
    )


def build_channel_sweep_series():
    """
    Physically-motivated one-at-a-time sweeps for train channel parameters.
    Design choices:
    - Keep MU_B_TR fixed at 0 (zero-mean additive noise).
    - Sweep fading mean/std and noise std.
    - Include a near-ideal point via tiny std values (exact 0 causes KL singularities).
    """
    eps_std = 1e-2
    sweep_spec = {
        # Fading mean around ideal mu_m=1.0 and test value MU_M_TE
        "mu_m_tr": [0.2, 0.4, MU_M_TE, 0.8, 1.0, 1.2, 1.6, 1.8, 2.0],
        # Fading variation from near-ideal to severe
        "std_m_tr": [eps_std, 0.05, 0.1, 0.2, 0.5, 0.75, STD_M_TE, 1.5, 2.0],
        # Additive noise std from near-ideal to severe
        "std_b_tr": [eps_std, 0.05, 0.1, 0.2, 0.5, 0.75, STD_B_TE, 1.5, 2.0],
    }
    configs = []
    for axis, values in sweep_spec.items():
        for v in values:
            cfg = {
                "axis": axis,
                "axis_value": float(v),
                "mu_m_tr": MU_M_TE,
                "std_m_tr": STD_M_TE,
                "mu_b_tr": 0.0,
                "std_b_tr": STD_B_TE,
            }
            cfg[axis] = float(v)
            configs.append(cfg)
    return configs, sweep_spec


def aggregate_channel_sweep_results(results):
    grouped = {}
    for r in results:
        key = (r["axis"], r["axis_value"])
        grouped.setdefault(key, []).append(r["acc"])

    summary = {}
    for key, accs in grouped.items():
        acc_arr = np.array(accs, dtype=np.float64)
        summary[key] = {
            "acc_mean": float(acc_arr.mean()),
            "acc_std": float(acc_arr.std(ddof=0)),
            "n": int(len(acc_arr)),
        }
    return summary


def plot_channel_sweep_results(results, sweep_spec):
    """
    Line plots: inference accuracy vs each swept train-channel parameter.
    """
    summary = aggregate_channel_sweep_results(results)
    axes_order = ["mu_m_tr", "std_m_tr", "std_b_tr"]
    test_values = {
        "mu_m_tr": MU_M_TE,
        "std_m_tr": STD_M_TE,
        "std_b_tr": STD_B_TE,
    }
    titles = {
        "mu_m_tr": "Sweep MU_M_TR",
        "std_m_tr": "Sweep STD_M_TR",
        "std_b_tr": "Sweep STD_B_TR",
    }

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    axs = np.atleast_1d(axs).ravel()

    for i, axis in enumerate(axes_order):
        x_vals = sorted(set(float(v) for v in sweep_spec[axis]))
        y_mean = [summary[(axis, x)]["acc_mean"] for x in x_vals]
        y_std = [summary[(axis, x)]["acc_std"] for x in x_vals]
        best_idx = int(np.argmax(y_mean))
        best_x, best_y = x_vals[best_idx], y_mean[best_idx]

        ax = axs[i]
        ax.errorbar(x_vals, y_mean, yerr=y_std, marker='o', capsize=3, label='Inference Acc')
        ax.axvline(test_values[axis], linestyle='--', color='tab:red', linewidth=1.5, label='Test-channel value')
        ax.scatter([best_x], [best_y], color='tab:green', s=70, zorder=4, label='Best in sweep')
        ax.set_title(titles[axis])
        ax.set_xlabel(axis)
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    fig.suptitle("Inference Accuracy vs Train-Channel Parameter (One-at-a-time Sweep)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(PLOTS_DIR, "channel_param_sweep_accuracy.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filename

# ==========================================
# 6. TRAINING LOOP & REGIMES
# ==========================================
def build_training_config(
    scenario_name,
    mode,
    objective,
    n_samples,
    reg_coeff,
    reg_alpha,
    reg_beta,
    mi_mc_samples,
    model_type,
    seed,
    lipschitz_method_perfect,
):
    return {
        "scenario": scenario_name,
        "mode": mode,
        "objective": objective,
        "n_samples": n_samples,
        "seed": seed,
        "dataset": {
            "name": "make_moons",
            "noise": MOONS_NOISE,
        },
        "model": {
            "type": model_type,
            "k_ensemble": K_ENSEMBLE if model_type == "VectorizedBNNEnsemble" else 1,
            "in_dim": IN_DIM,
            "hid_dim": HIDDEN_DIM,
            "out_dim": OUT_DIM,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR_BASE,
            "lr_decay_step": LR_DECAY_STEP,
            "lr_decay_gamma": LR_DECAY_GAMMA,
            "prior_lambda": PRIOR_LAMBDA,
            "epsilon": EPSILON,
            "sigma_sq": SIGMA_SQ,
            "sigma_0_sq": SIGMA_0_SQ,
            "reg_coeff": reg_coeff,
            "reg_alpha": reg_alpha,
            "reg_beta": reg_beta,
            "mi_mc_samples": mi_mc_samples,
            "lipschitz_method_perfect": lipschitz_method_perfect,
        },
        "channel_train": {
            "mu_m": MU_M_TR,
            "std_m": STD_M_TR,
            "mu_b": MU_B_TR,
            "std_b": STD_B_TR,
        },
        "channel_test": {
            "mu_m": MU_M_TE,
            "std_m": STD_M_TE,
            "mu_b": MU_B_TE,
            "std_b": STD_B_TE,
        },
    }


def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _normalize_reg_params_for_weights(scenario_name, objective, reg_coeff, reg_alpha, reg_beta):
    """
    Normalize regularization parameters so weights are cached only when they
    actually affect training.
    """
    if scenario_name == 'erm':
        return None, None, None
    if scenario_name == 'pac_bayes':
        return reg_coeff, None, None
    if scenario_name == 'proposed':
        if objective == 'bound':
            return reg_coeff, None, None
        if objective == 'heuristic':
            return None, reg_alpha, reg_beta
    return reg_coeff, reg_alpha, reg_beta


def _normalize_mi_samples_for_weights(scenario_name, mode, mi_mc_samples):
    if scenario_name == 'proposed' and mode == 'train':
        return mi_mc_samples
    return None


def _normalize_channel_test_for_weights(scenario_name):
    if scenario_name == 'proposed':
        return {
            "mu_m": MU_M_TE,
            "std_m": STD_M_TE,
            "mu_b": MU_B_TE,
            "std_b": STD_B_TE,
        }
    return None


def get_weights_path(
    scenario_name,
    mode,
    objective,
    n_samples,
    reg_coeff,
    reg_alpha,
    reg_beta,
    mi_mc_samples,
    model_type,
    seed,
    lipschitz_method_perfect,
):
    eff_coeff, eff_alpha, eff_beta = _normalize_reg_params_for_weights(
        scenario_name,
        objective,
        reg_coeff,
        reg_alpha,
        reg_beta,
    )
    eff_mi_samples = _normalize_mi_samples_for_weights(scenario_name, mode, mi_mc_samples)
    cfg = build_training_config(
        scenario_name,
        mode,
        objective,
        n_samples,
        eff_coeff,
        eff_alpha,
        eff_beta,
        eff_mi_samples,
        model_type,
        seed,
        lipschitz_method_perfect,
    )
    cfg["channel_test"] = _normalize_channel_test_for_weights(scenario_name)
    run_id = config_hash(cfg)
    filename = f"weights_{scenario_name}_{mode}_{objective}_{run_id}.pth"
    return os.path.join(WEIGHTS_DIR, filename), cfg


def train_scenario(
    scenario_name,
    loader,
    n_samples,
    mode='perfect',
    objective='bound',
    reg_coeff=None,
    reg_alpha=None,
    reg_beta=None,
    mi_mc_samples=None,
    seed=None,
    lipschitz_method_perfect=None,
    use_cache=True,
    plot_suffix=None,
    plot_title_extra=None,
):
    print(f"\n--- Training Scenario: {scenario_name.upper()} ---")
    if scenario_name == 'erm':
        model = DeterministicFC(IN_DIM, HIDDEN_DIM, OUT_DIM).to(device)
        model_type = "DeterministicFC"
    else:
        model = VectorizedBNNEnsemble(K_ENSEMBLE, IN_DIM, HIDDEN_DIM, OUT_DIM).to(device)
        model_type = "VectorizedBNNEnsemble"
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_BASE)
    scheduler = None
    if LR_DECAY_STEP and LR_DECAY_STEP > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=LR_DECAY_STEP,
            gamma=LR_DECAY_GAMMA,
        )

    reg_coeff = REG_COEFF if reg_coeff is None else reg_coeff
    reg_alpha = REG_ALPHA if reg_alpha is None else reg_alpha
    reg_beta = REG_BETA if reg_beta is None else reg_beta
    mi_mc_samples = MI_MC_SAMPLES if mi_mc_samples is None else mi_mc_samples
    lipschitz_method_perfect = LIPSCHITZ_METHOD_PERFECT if lipschitz_method_perfect is None else lipschitz_method_perfect
    if lipschitz_method_perfect not in {"grad", "analytical"}:
        raise ValueError("lipschitz_method_perfect must be 'grad' or 'analytical'.")
    lipschitz_method_for_cache = (
        lipschitz_method_perfect if (mode == "perfect" and scenario_name == "proposed") else None
    )

    seed = SEED if seed is None else seed
    weights_path, run_cfg = get_weights_path(
        scenario_name,
        mode,
        objective,
        n_samples,
        reg_coeff,
        reg_alpha,
        reg_beta,
        mi_mc_samples,
        model_type,
        seed,
        lipschitz_method_for_cache,
    )
    if use_cache and os.path.exists(weights_path):
        print(f"Loading existing weights: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        return model
    
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
        epoch_kl_ch_total = 0.0
        epoch_expected_kl = 0.0
        epoch_k_hat = 0.0
        epoch_channel_penalty = 0.0
        epoch_k_hat_channel_penalty = 0.0

        model.train()
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            if getattr(model, "is_bnn", False):
                # Forward pass (1 MC sample for speed during CE calculation)
                theta = model.sample_theta(1).squeeze(0)
                # Pass mode='train' to sample from the seeded P_art sequence
                out = model(batch_x, theta, mode=mode)

                # Cross Entropy Loss
                y_expanded = batch_y.unsqueeze(0).expand(model.K, -1)
                ce_loss = F.cross_entropy(out.reshape(-1, OUT_DIM), y_expanded.reshape(-1))
            else:
                logits = model(batch_x, mode=mode)
                ce_loss = F.cross_entropy(logits, batch_y)
            
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

            if (scenario_name == 'proposed' and mode == 'perfect'):
                expected_kl = compute_expected_kl(model)
                expected_kl_val = expected_kl.item()

                # Lipschitz
                if lipschitz_method_perfect == "grad":
                    grad_theta = torch.autograd.grad(
                        ce_loss,
                        theta,
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    K_hat = torch.norm(grad_theta, p=2)
                else:
                    K_hat = model.compute_analytical_lipschitz(batch_x, theta, mode=mode)
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
                    loss = ce_loss + (reg_coeff * reg)
                elif objective == 'heuristic':
                    loss = ce_loss + (reg_alpha * K_hat * CH_PENALTY + reg_beta * expected_kl)
            
            if scenario_name == 'pac_bayes':
                # MC Sampling for bounds
                kl_mix, _ = compute_mixture_kl_and_mi(model, num_samples=mi_mc_samples)

                # Standard Regularization Formula
                reg = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (torch.clamp(kl_mix, min=0) + complexity_term))
                loss = ce_loss + (reg_coeff * reg)
                reg_val = reg.item()
                term1_val = reg.item()
                

            if scenario_name == 'proposed' and mode == 'train':
                expected_kl = compute_expected_kl(model)
                expected_kl_val = expected_kl.item()

                # MC Sampling for bounds
                _, mi = compute_mixture_kl_and_mi(model, num_samples=mi_mc_samples)

                # Proposed Framework Formula
                term1 = torch.sqrt(2 * SIGMA_0_SQ * (torch.clamp(mi, min=0) + KL_CH_TOTAL))
                term2 = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (expected_kl + complexity_term))
                reg = term1 + term2
                term1_val = term1.item()
                term2_val = term2.item()
                mi_val = mi.item()
                kl_ch_total_val = KL_CH_TOTAL
                
                if objective == 'bound':
                    loss = ce_loss + (reg_coeff * reg)
                elif objective == 'heuristic':
                    loss = ce_loss + (reg_alpha * mi + reg_beta * expected_kl)
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
        history['mi'].append(epoch_mi / len(loader))
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
        if scheduler is not None:
            scheduler.step()

    print(
        f"Epoch {EPOCHS}/{EPOCHS} | "
        f"CE Loss: {epoch_loss/len(loader):.4f} | "
        f"Reg Bound: {epoch_reg/len(loader):.4f} | "
        f"Bound T1: {epoch_term1/len(loader):.4f} | "
        f"Bound T2: {epoch_term2/len(loader):.4f} | "
        f"Train Acc: {train_acc*100:.2f}%"
    )

    if use_cache:
        torch.save(model.state_dict(), weights_path)
        cfg_path = weights_path.replace(".pth", ".json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, indent=2)
    plot_training_metrics(history, scenario_name, mode, plot_suffix=plot_suffix, title_extra=plot_title_extra)
    plot_bound_decomposition(history, scenario_name, mode, plot_suffix=plot_suffix, title_extra=plot_title_extra)
            
    return model

# ==========================================
# 7. EVALUATION ON INFERENCE CHANNEL (P_ch)
# ==========================================
def evaluate_inference(model, loader, repeats=1, weight_mc_samples=1):
    model.eval()

    loss_runs = []
    acc_runs = []

    with torch.no_grad():
        # # Use posterior mean if weight_mc_samples == 1, else sample weights.
        # theta_mean = model.mu if getattr(model, "is_bnn", False) else None

        for _ in range(repeats):
            total_loss = 0.0
            total_correct = 0
            total = 0

            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Monte Carlo over weight samples at test time (channel sampled inside forward)
                logits_sum = None
                probs_sum = None
                total_mc = 0

                if getattr(model, "is_bnn", False):
                    for _ in range(weight_mc_samples):

                        theta = model.sample_theta(1).squeeze(0)

                        out = model(batch_x, theta, mode='test')
                        logits = out.mean(dim=0)
                        probs = F.softmax(out, dim=-1).mean(dim=0)

                        if logits_sum is None:
                            logits_sum = logits
                            probs_sum = probs
                        else:
                            logits_sum = logits_sum + logits
                            probs_sum = probs_sum + probs
                        total_mc += 1
                else:
                    logits = model(batch_x, mode='test')
                    probs = F.softmax(logits, dim=-1)

                    logits_sum = logits
                    probs_sum = probs
                    total_mc = 1

                avg_logits = logits_sum / max(total_mc, 1)
                avg_probs = probs_sum / max(total_mc, 1)
                loss = F.cross_entropy(avg_logits, batch_y)
                preds = avg_probs.argmax(dim=-1)

                total_loss += loss.item() * batch_y.size(0)
                total_correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            loss_runs.append(total_loss / max(total, 1))
            acc_runs.append(total_correct / max(total, 1))

    return float(np.mean(loss_runs)), float(np.mean(acc_runs))

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    RUN_CHANNEL_PARAM_SWEEP = True
    EVAL_REPEATS = 10
    INFERENCE_WEIGHT_SAMPLES = 10
    CHANNEL_SWEEP_SEEDS = [SEED]
    CHANNEL_SWEEP_SCENARIO = "proposed"  # Options: "erm", "proposed"
    CHANNEL_SWEEP_OBJECTIVE = "bound"  # Used when CHANNEL_SWEEP_SCENARIO == "proposed"

    if RUN_CHANNEL_PARAM_SWEEP:
        original_train_channel = (MU_M_TR, STD_M_TR, MU_B_TR, STD_B_TR)
        channel_configs, sweep_spec = build_channel_sweep_series()
        sweep_results = []

        for seed in CHANNEL_SWEEP_SEEDS:
            set_seed(seed)
            train_loader, test_loader, n_trains = get_dataloaders(
                N_SAMPLES,
                batch_size=BATCH_SIZE,
                seed=seed,
                noise=MOONS_NOISE,
            )

            for cfg in channel_configs:
                set_train_channel_params(
                    cfg["mu_m_tr"],
                    cfg["std_m_tr"],
                    cfg["mu_b_tr"],
                    cfg["std_b_tr"],
                )
                print(
                    f"Sweeping axis={cfg['axis']} value={cfg['axis_value']:.4f} "
                    f"(seed={seed}, scenario={CHANNEL_SWEEP_SCENARIO})"
                )
                plot_suffix = build_channel_plot_label(cfg, seed)
                plot_title_extra = build_channel_title(cfg)

                model = train_scenario(
                    CHANNEL_SWEEP_SCENARIO,
                    train_loader,
                    n_trains,
                    mode='train',
                    objective=CHANNEL_SWEEP_OBJECTIVE,
                    seed=seed,
                    use_cache=False,
                    plot_suffix=plot_suffix,
                    plot_title_extra=plot_title_extra,
                )
                loss, acc = evaluate_inference(
                    model,
                    test_loader,
                    repeats=EVAL_REPEATS,
                    weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
                )
                sweep_results.append({
                    "seed": seed,
                    "scenario": CHANNEL_SWEEP_SCENARIO,
                    "objective": CHANNEL_SWEEP_OBJECTIVE,
                    "mode": "train",
                    "axis": cfg["axis"],
                    "axis_value": cfg["axis_value"],
                    "mu_m_tr": cfg["mu_m_tr"],
                    "std_m_tr": cfg["std_m_tr"],
                    "mu_b_tr": cfg["mu_b_tr"],
                    "std_b_tr": cfg["std_b_tr"],
                    "kl_ch_total": float(KL_CH_TOTAL),
                    "loss": loss,
                    "acc": acc
                })

        # Restore original train-channel parameters after sweep
        set_train_channel_params(*original_train_channel)

        summary = aggregate_channel_sweep_results(sweep_results)
        best_entry = max(
            summary.items(),
            key=lambda kv: kv[1]["acc_mean"]
        )
        (best_axis, best_value), best_stats = best_entry

        results_path = os.path.join(RESULTS_DIR, "channel_param_sweep_results.json")
        summary_path = os.path.join(RESULTS_DIR, "channel_param_sweep_summary.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2)
        serializable_summary = []
        for (axis, axis_value), stats in summary.items():
            serializable_summary.append({
                "axis": axis,
                "axis_value": axis_value,
                "acc_mean": stats["acc_mean"],
                "acc_std": stats["acc_std"],
                "n": stats["n"],
            })
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(serializable_summary, f, indent=2)

        plot_file = plot_channel_sweep_results(sweep_results, sweep_spec)

        print("\n" + "="*60)
        print("CHANNEL PARAMETER SWEEP COMPLETE")
        print("="*60)
        print(f"Results saved to: {results_path}")
        print(f"Summary saved to: {summary_path}")
        print(f"Plot saved to:    {plot_file}")
        print(
            f"Best axis/value: {best_axis}={best_value:.4f} "
            f"(mean acc={best_stats['acc_mean']*100:.2f}%, std={best_stats['acc_std']*100:.2f}%)"
        )
        print("="*60)
    else:
        set_seed(SEED)
        train_loader, test_loader, n_trains = get_dataloaders(
            N_SAMPLES,
            batch_size=BATCH_SIZE,
            noise=MOONS_NOISE,
        )

        # Scenario A: Standard ERM + perfect channel
        model_erm_perfect = train_scenario('erm', train_loader, n_trains, mode='perfect', objective='bound', seed=SEED)
        loss_erm_perfect, acc_erm_perfect = evaluate_inference(
            model_erm_perfect,
            test_loader,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
        )

        # Scenario B: Standard ERM + train channel (overfitting to P_art)
        model_erm = train_scenario('erm', train_loader, n_trains, mode='train', objective='bound', seed=SEED)
        loss_erm, acc_erm = evaluate_inference(
            model_erm,
            test_loader,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
        )

        # Scenario C: Proposed Bound Regularization
        model_prop = train_scenario('proposed', train_loader, n_trains, mode='train', objective='bound', seed=SEED)
        loss_prop, acc_prop = evaluate_inference(
            model_prop,
            test_loader,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
        )


        # Scenario D: Proposed Bound + perfect channel
        model_prop_perfect = train_scenario('proposed', train_loader, n_trains, mode='perfect', objective='heuristic', seed=SEED)
        loss_prop_perfect, acc_prop_perfect = evaluate_inference(
            model_prop_perfect,
            test_loader,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
        )


        print("\n" + "="*50)
        print("FINAL INFERENCE RESULTS (EVALUATED ON P_ch)")
        print("="*50)
        print(f"Standard ERM + Perfect Channel Loss/Acc: {loss_erm_perfect:.4f} / {acc_erm_perfect*100:.2f}%")
        print(f"Standard ERM Loss/Acc:                {loss_erm:.4f} / {acc_erm*100:.2f}%")
        print(f"Proposed Bound Reg Loss/Acc:          {loss_prop:.4f} / {acc_prop*100:.2f}%")
        print(f"Proposed Bound + Perfect Channel Loss/Acc: {loss_prop_perfect:.4f} / {acc_prop_perfect*100:.2f}%")
        print("="*50)
