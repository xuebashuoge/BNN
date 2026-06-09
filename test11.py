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


# ERM using deterministic / proposed using new results Theorem 4 (given S), revise the sweeping

# ==========================================
# 0. OUTPUT PATHS
# ==========================================
RESULTS_DIR = os.path.join('results', 'test11')
os.makedirs(RESULTS_DIR, exist_ok=True)

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
N_U_SETS = 10          # Number of independently sampled U sets / posterior components
HIDDEN_DIM = 64       # INCREASED: Give ERM enough capacity to overfit the channel
IN_DIM = 2
OUT_DIM = 2
MOONS_NOISE = 0.3      # make_moons noise level
BATCH_SIZE = 64        # Reduced batch size for noisier gradients
EPOCHS = 150           # Increased epochs to ensure ERM fully memorizes
LR_BASE = 0.001         # Adjusted base learning rate
LR_DECAY_STEP = 0     # StepLR decay period (epochs)
LR_DECAY_GAMMA = 0.5   # StepLR decay factor
PRIOR_LAMBDA = 1.0     # Variance of isotropic Gaussian prior
EPSILON = 0.025         # PAC-Bayes confidence parameter
SIGMA_SQ = 1.0         # Assumed sub-Gaussian parameter
SIGMA_ART_SQ = 1.0     # Assumed sub-Gaussian parameter for artificial-channel loss
ALPHA_COEFF = 0.0      # Weighting factor for the channel-shifting term in the objective
BETA_COEFF = 0.19976502246972636       # Weighting factor for the channel-overfitting term in the objective
GAMMA_COEFF = 0.05989606713587279     # Weighting factor for the standard PAC-Bayes term in the objective (optional ablation)
M_ARTIFICIAL_CHANNELS = 100  # m: size of the fixed artificial channel set U
MI_MC_SAMPLES = 100       # MC samples for mixture KL / channel-overfitting estimation
SEED = 5
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

# Closed form channel-shift KL D(P_ch || P_art)
def channel_kl_total(hidden_dim):
    kl_ch_m = hidden_dim * kl_gaussian_1d(MU_M_TE, STD_M_TE, MU_M_TR, STD_M_TR)
    kl_ch_b = hidden_dim * kl_gaussian_1d(MU_B_TE, STD_B_TE, MU_B_TR, STD_B_TR)
    return kl_ch_m + kl_ch_b

KL_CH_TOTAL = channel_kl_total(HIDDEN_DIM)
print(f"Default D(P_ch || P_art) = {KL_CH_TOTAL:.4f}")

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

CHANNEL_PENALTY_CACHE = {}

def get_channel_penalty(hidden_dim):
    if hidden_dim not in CHANNEL_PENALTY_CACHE:
        CHANNEL_PENALTY_CACHE[hidden_dim] = estimate_expected_channel_norm(
            hidden_dim,
            MU_M_TE,
            MU_B_TE,
            STD_M_TE,
            STD_B_TE,
            norm_type='frobenius',
            device=device,
        )
    return CHANNEL_PENALTY_CACHE[hidden_dim]

# channel penalty for default single-run config
CH_PENALTY = get_channel_penalty(HIDDEN_DIM)


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
    During training, component k samples from its own fixed artificial channel set U_k.
    During inference, samples fresh channel weights from P_ch.
    """
    def __init__(self, K, hid_dim, num_artificial_channels):
        super().__init__()
        self.K = K
        self.hid_dim = hid_dim
        self.num_artificial_channels = num_artificial_channels
        self.train_gen = torch.Generator(device='cpu')
        self.train_gen.manual_seed(1337)
        u_m = torch.randn(K, num_artificial_channels, hid_dim, generator=self.train_gen) * STD_M_TR + MU_M_TR
        u_b = torch.randn(K, num_artificial_channels, hid_dim, generator=self.train_gen) * STD_B_TR + MU_B_TR
        self.register_buffer("u_m", u_m)
        self.register_buffer("u_b", u_b)
        self.last_m = None

    def forward(self, x, mode='perfect'):
        # x shape: (K, Batch, hid_dim)
        B = x.shape[1]
        
        if mode == 'perfect':
            # No channel effect, perfect transmission
            m = torch.ones_like(x, device=x.device)
            b = torch.zeros_like(x, device=x.device)
        elif mode == 'train':
            # Sample with replacement from each component's own U_k.
            idx = torch.randint(
                self.num_artificial_channels,
                (self.K, B),
                generator=self.train_gen,
                device='cpu',
            ).to(x.device)
            component_idx = torch.arange(self.K, device=x.device).unsqueeze(1)
            m = self.u_m.to(x.device)[component_idx, idx]
            b = self.u_b.to(x.device)[component_idx, idx]
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
    Vectorized representation of K posterior components.
    Component k is trained with an independent artificial channel set U_k,
    so the mixture over k approximates P_{tilde W|S} = E_U[P_{tilde W|U,S}].
    """
    def __init__(self, K, in_dim, hid_dim, out_dim, num_artificial_channels):
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
        
        # Learnable posterior parameters for each sampled U_k.
        self.mu = nn.Parameter(torch.randn(K, self.D) * 0.1)
        self.rho = nn.Parameter(torch.randn(K, self.D) * 0.1 - 3.0)
        
        # Instantiate the stochastic channel layer
        self.channel_layer = StochasticChannelLayer(K, hid_dim, num_artificial_channels)

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
    def __init__(self, in_dim, hid_dim, out_dim, num_artificial_channels):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.channel_layer = StochasticChannelLayer(1, hid_dim, num_artificial_channels)
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

def compute_component_expected_kl(model):
    """Calculates E_U[ D(P_{tilde W|U,S} || Q) ] over sampled U_k components."""
    mu = model.mu
    sigma = model.get_sigma()
    var = sigma**2
    # Analytical KL per sampled U_k component.
    kl_k = 0.5 * torch.sum(var/PRIOR_LAMBDA + mu**2/PRIOR_LAMBDA - 1 - torch.log(var/PRIOR_LAMBDA), dim=-1)
    return kl_k.mean()

def compute_mixture_kl_and_channel_overfit(model, num_samples=3):
    """
    Approximates the intractable terms via Monte Carlo sampling.
    Returns:
    - D(P_{tilde W|S} || Q_{tilde W}) where P_{tilde W|S} is a mixture.
    - D(P_{tilde W U|S} || P_U P_{tilde W|S}) as a channel-overfitting proxy, equals to E_{P_UP_{tilde W|U,S}} [log P_{tilde W|U,S} - log P_{tilde W|S}]
    The K mixture components correspond to independent sampled channel sets U_k.
    """
    theta = model.sample_theta(num_samples) # Shape: (S, K, D)
    mu = model.mu
    sigma = model.get_sigma()
    
    # 1. Log density of samples under their own P_{tilde W|U_k,S}.
    log_q_k = log_gaussian(theta, mu, sigma) # Shape: (S, K)
    
    # 2. Log density of samples under the mixture marginal P_{tilde W|S}.
    # Expand dims to compute all pairwise densities: N(theta_{s,k} | mu_j, sigma_j)
    theta_exp = theta.unsqueeze(2)        # (S, K, 1, D)
    mu_exp = mu.unsqueeze(0).unsqueeze(0) # (1, 1, K_eval, D)
    sigma_exp = sigma.unsqueeze(0).unsqueeze(0) 
    
    var_exp = sigma_exp**2 + 1e-8
    log_scale = torch.log(var_exp) * 0.5 + 0.5 * np.log(2 * np.pi)
    diff = (theta_exp - mu_exp)**2 / (2 * var_exp)

    # sum 1/sqrt(2*pi*var) + exp(-diff) over K_eval components to get P_{tilde W|S} for each sample, log P_{tilde W|S} = log(sum_j -1/2 log(2*pi*var_j) - diff_j))
    log_q_all = -torch.sum(log_scale + diff, dim=-1) # Shape: (S, K, K_eval)
    
    # Use logsumexp for numerical stability, log P_{tilde W|S} = log(sum_j exp(log P(tilde W|U_j,S))) - log(K)
    log_p_mix = torch.logsumexp(log_q_all, dim=2) - np.log(model.K) # Shape: (S, K)
    
    # Channel overfitting: E[log P_{tilde W|U,S} - log P_{tilde W|S}]
    channel_overfit = (log_q_k - log_p_mix).mean()
    
    # Standard KL: E_{tilde W~P_mix} [log P_{tilde W|S} - log Q]
    log_q_prior = log_prior(theta, PRIOR_LAMBDA) # Shape: (S, K)
    kl_mix = (log_p_mix - log_q_prior).mean()
    
    return kl_mix, channel_overfit


def evaluate_model(model, loader, mode='train'):
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
                # For BNN, average loss rather than logits
                out = model(batch_x, theta_mean, mode=mode)
                
                # Calculate expected loss for every K component individually to mirror the true mathematical formula
                y_expanded = batch_y.unsqueeze(0).expand(model.K, -1)
                loss = F.cross_entropy(out.reshape(-1, model.out_dim), y_expanded.reshape(-1))

                preds = out.argmax(dim=-1) # Shape: (K, Batch)

                total_correct += (preds == y_expanded).sum().item() / model.K  # Average correct predictions across K components

                # avg_logits = out.mean(dim=0)
                # loss = F.cross_entropy(avg_logits, batch_y)

                # probs = F.softmax(out, dim=-1)
                # avg_probs = probs.mean(dim=0)
                # preds = avg_probs.argmax(dim=-1)
            else:
                logits = model(batch_x, mode=mode)
                loss = F.cross_entropy(logits, batch_y)
                preds = logits.argmax(dim=-1)
                total_correct += (preds == batch_y).sum().item()

            total_loss += loss.item() * batch_y.size(0)
            total += batch_y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def scenario_label(scenario_name, mode, objective):
    if scenario_name in {"erm"}:
        return f"{scenario_name}_{mode}"
    return f"{scenario_name}_{mode}_{objective}"


def plot_training_metrics(history, scenario_name, mode, objective, run_dir):
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

    filename = os.path.join(run_dir, "metrics.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_bound_decomposition(history, scenario_name, mode, objective, run_dir):
    """Plots and saves bound decomposition curves if available."""
    if not history.get('bound_total'):
        return
    if max(history['bound_total']) == 0.0 and max(history['bound_term1']) == 0.0 and max(history['bound_term2']) == 0.0:
        return

    epochs = list(range(1, len(history['bound_total']) + 1))
    series = [
        ('Bound Total', history['bound_total']),
        ('Channel Shifting', history['bound_term1']),
        ('Channel Overfitting', history['bound_term2']),
        ('Standard PAC-Bayes', history.get('bound_term3', [])),
        ('Channel Overfit KL', history.get('channel_overfit_kl', [])),
        ('D(P_ch || P_art)', history.get('kl_ch_total', [])),
        ('Mixture KL', history.get('mixture_kl', [])),
        ('Component E[KL]', history.get('component_expected_kl', [])),
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

    filename = os.path.join(run_dir, "bound.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 6. TRAINING LOOP & REGIMES
# ==========================================
def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

def build_param_config(
    scenario_name,
    mode,
    objective,
    n_samples,
    seed,
    mi_mc_samples,
    lipschitz_method_perfect,
    lr=None,
    alpha_coeff=None,
    beta_coeff=None,
    gamma_coeff=None,
    hidden_dim=None,
    n_u_sets=None,
    m_artificial_channels=None,
):
    lr = LR_BASE if lr is None else lr
    alpha_coeff = ALPHA_COEFF if alpha_coeff is None else alpha_coeff
    beta_coeff = BETA_COEFF if beta_coeff is None else beta_coeff
    gamma_coeff = GAMMA_COEFF if gamma_coeff is None else gamma_coeff
    hidden_dim = HIDDEN_DIM if hidden_dim is None else hidden_dim
    n_u_sets = N_U_SETS if n_u_sets is None else n_u_sets
    m_artificial_channels = M_ARTIFICIAL_CHANNELS if m_artificial_channels is None else m_artificial_channels
    return {
        "scenario_name": scenario_name,
        "mode": mode,
        "objective": "N/A" if scenario_name == 'erm' else objective,
        "n_samples_total": N_SAMPLES,
        "n_samples_train": n_samples,
        "seed": seed,
        "dataset": {
            "name": "make_moons",
            "noise": MOONS_NOISE,
        },
        "model": {
            "in_dim": IN_DIM,
            "hid_dim": hidden_dim,
            "out_dim": OUT_DIM,
            "n_u_sets": n_u_sets,
            "posterior_components": n_u_sets,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": lr,
            "lr_decay_step": LR_DECAY_STEP,
            "lr_decay_gamma": LR_DECAY_GAMMA,
            "prior_lambda": PRIOR_LAMBDA,
            "epsilon": EPSILON,
            "sigma_sq": SIGMA_SQ,
            "sigma_art_sq": SIGMA_ART_SQ,
            "alpha_coeff": alpha_coeff if scenario_name != 'erm' else "N/A",
            "beta_coeff": beta_coeff if scenario_name != 'erm' else "N/A",
            "gamma_coeff": gamma_coeff if scenario_name != 'erm' else "N/A",
            "mi_mc_samples": mi_mc_samples,
            "m_artificial_channels": m_artificial_channels,
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


def get_run_dir(
    scenario_name,
    mode,
    objective,
    n_samples,
    seed,
    mi_mc_samples,
    lipschitz_method_perfect,
    lr=None,
    alpha_coeff=None,
    beta_coeff=None,
    gamma_coeff=None,
    hidden_dim=None,
    n_u_sets=None,
    m_artificial_channels=None,
):
    param_cfg = build_param_config(
        scenario_name,
        mode,
        objective,
        n_samples,
        seed,
        mi_mc_samples,
        lipschitz_method_perfect,
        lr=lr,
        alpha_coeff=alpha_coeff,
        beta_coeff=beta_coeff,
        gamma_coeff=gamma_coeff,
        hidden_dim=hidden_dim,
        n_u_sets=n_u_sets,
        m_artificial_channels=m_artificial_channels,
    )
    run_id = config_hash(param_cfg)
    label = scenario_label(scenario_name, mode, objective)

    run_dir = os.path.join(RESULTS_DIR, label, f"param_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    params_path = os.path.join(run_dir, "params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(param_cfg, f, indent=2)
        
    return run_dir


def is_converged_history(history):
    if not history.get("train_loss") or not history.get("train_acc"):
        return False

    final_loss = history["train_loss"][-1]
    final_acc = history["train_acc"][-1]
    finite_series = all(
        np.isfinite(v)
        for key in ("train_loss", "bound_total", "bound_term1", "bound_term2", "bound_term3")
        for v in history.get(key, [])
    )
    if not finite_series or not np.isfinite(final_loss) or not np.isfinite(final_acc):
        return False
    if final_acc < 0.55:
        return False
    if final_loss > 5.0:
        return False
    return True


def save_training_history(
    run_dir,
    scenario_name,
    mode,
    objective,
    history,
    lr,
    alpha_coeff,
    beta_coeff,
    gamma_coeff,
    inference=None,
):
    payload = {
        "scenario": scenario_name,
        "mode": mode,
        "objective": objective,
        "lr": lr,
        "alpha": alpha_coeff,
        "beta": beta_coeff,
        "gamma": gamma_coeff,
        "converged": is_converged_history(history),
        "final_train_loss": history["train_loss"][-1] if history.get("train_loss") else None,
        "final_train_acc": history["train_acc"][-1] if history.get("train_acc") else None,
        "final_bound_total": history["bound_total"][-1] if history.get("bound_total") else None,
        "inference": inference,
        "history": history,
    }
    path = os.path.join(run_dir, "history.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def update_run_history_inference(run_dir, inference):
    path = os.path.join(run_dir, "history.json")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["inference"] = inference
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def estimate_term_scales(
    loader,
    n_samples,
    mi_mc_samples,
    seed=None,
    max_batches=1,
    hidden_dim=None,
    n_u_sets=None,
    m_artificial_channels=None,
):
    """
    Estimate scale of CE loss and bound terms for heuristic coefficient selection.
    Uses an untrained BNN and a few batches to get typical magnitudes.
    """
    if seed is not None:
        set_seed(seed)

    hidden_dim = HIDDEN_DIM if hidden_dim is None else hidden_dim
    n_u_sets = N_U_SETS if n_u_sets is None else n_u_sets
    m_artificial_channels = M_ARTIFICIAL_CHANNELS if m_artificial_channels is None else m_artificial_channels
    kl_ch_total = channel_kl_total(hidden_dim)
    model = VectorizedBNNEnsemble(n_u_sets, IN_DIM, hidden_dim, OUT_DIM, m_artificial_channels).to(device)
    model.eval()

    ce_losses = []
    channel_shift_vals = []
    channel_overfit_vals = []
    model_complexity_vals = []

    complexity_term = np.log(np.sqrt(n_samples) / EPSILON)
    denom = max(n_samples - 1, 1)

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            theta = model.sample_theta(1).squeeze(0)
            out = model(batch_x, theta, mode='train')
            y_expanded = batch_y.unsqueeze(0).expand(model.K, -1)
            ce_loss = F.cross_entropy(out.reshape(-1, OUT_DIM), y_expanded.reshape(-1))
            ce_losses.append(ce_loss.item())

            mixture_kl, channel_overfit_kl = compute_mixture_kl_and_channel_overfit(
                model,
                num_samples=mi_mc_samples,
            )
            mixture_kl = torch.clamp(mixture_kl, min=0)
            channel_overfit_kl = torch.clamp(channel_overfit_kl, min=0)

            channel_shift = torch.sqrt(ce_loss.new_tensor(2 * SIGMA_ART_SQ * kl_ch_total))
            channel_overfit = torch.sqrt((2 * SIGMA_ART_SQ / m_artificial_channels) * channel_overfit_kl)
            model_complexity = torch.sqrt((2 * SIGMA_SQ / denom) * (mixture_kl + complexity_term))

            channel_shift_vals.append(channel_shift.item())
            channel_overfit_vals.append(channel_overfit.item())
            model_complexity_vals.append(model_complexity.item())

    def safe_mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "ce_loss": safe_mean(ce_losses),
        "channel_shift": safe_mean(channel_shift_vals),
        "channel_overfit": safe_mean(channel_overfit_vals),
        "model_complexity": safe_mean(model_complexity_vals),
    }


def train_scenario(
    scenario_name,
    loader,
    n_samples,
    mode='perfect',
    objective='bound',
    lr=None,
    alpha_coeff=None,
    beta_coeff=None,
    gamma_coeff=None,
    mi_mc_samples=None,
    seed=None,
    lipschitz_method_perfect=None,
    hidden_dim=None,
    n_u_sets=None,
    m_artificial_channels=None,
    use_cache=True,
    run_dir=None,
):
    print(f"\n--- Training Scenario: {scenario_name.upper()} ---")
    lr = LR_BASE if lr is None else lr
    alpha_coeff = ALPHA_COEFF if alpha_coeff is None else alpha_coeff
    beta_coeff = BETA_COEFF if beta_coeff is None else beta_coeff
    gamma_coeff = GAMMA_COEFF if gamma_coeff is None else gamma_coeff
    hidden_dim = HIDDEN_DIM if hidden_dim is None else hidden_dim
    n_u_sets = N_U_SETS if n_u_sets is None else n_u_sets
    m_artificial_channels = M_ARTIFICIAL_CHANNELS if m_artificial_channels is None else m_artificial_channels
    kl_ch_total = channel_kl_total(hidden_dim)
    channel_penalty = get_channel_penalty(hidden_dim)
    seed = SEED if seed is None else seed
    set_seed(seed)

    if scenario_name in {'erm', 'l2'}:
        model = DeterministicFC(IN_DIM, hidden_dim, OUT_DIM, m_artificial_channels).to(device)
    else:
        model = VectorizedBNNEnsemble(n_u_sets, IN_DIM, hidden_dim, OUT_DIM, m_artificial_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if LR_DECAY_STEP and LR_DECAY_STEP > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=LR_DECAY_STEP,
            gamma=LR_DECAY_GAMMA,
        )

    mi_mc_samples = MI_MC_SAMPLES if mi_mc_samples is None else mi_mc_samples
    lipschitz_method_perfect = LIPSCHITZ_METHOD_PERFECT if lipschitz_method_perfect is None else lipschitz_method_perfect
    if lipschitz_method_perfect not in {"grad", "analytical"}:
        raise ValueError("lipschitz_method_perfect must be 'grad' or 'analytical'.")
    if run_dir is None and use_cache:
        run_dir = get_run_dir(
            scenario_name, mode, objective,
            n_samples, seed, mi_mc_samples, lipschitz_method_perfect,
            lr=lr, alpha_coeff=alpha_coeff, beta_coeff=beta_coeff, gamma_coeff=gamma_coeff,
            hidden_dim=hidden_dim, n_u_sets=n_u_sets, m_artificial_channels=m_artificial_channels,
        )
    else:
        run_dir = run_dir or "temp_run"

    weights_path = os.path.join(run_dir, "weights.pth")

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
        'bound_term3': [],
        'channel_overfit_kl': [],
        'kl_ch_total': [],
        'mixture_kl': [],
        'component_expected_kl': [],
        'k_hat': [],
        'channel_penalty': [],
        'k_hat_channel_penalty': [],
        'applied_regularization': [],
        'nonfinite_loss': False,
    }
    stop_training = False

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_reg = 0.0
        epoch_bound_total = 0.0
        epoch_term1 = 0.0
        epoch_term2 = 0.0
        epoch_term3 = 0.0
        epoch_channel_overfit_kl = 0.0
        epoch_kl_ch_total = 0.0
        epoch_mixture_kl = 0.0
        epoch_component_expected_kl = 0.0
        epoch_k_hat = 0.0
        epoch_channel_penalty = 0.0
        epoch_k_hat_channel_penalty = 0.0

        model.train()
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            if getattr(model, "is_bnn", False):
                # For BNN
                # Forward pass (1 MC sample for speed during CE calculation)
                theta = model.sample_theta(1).squeeze(0)
                # Pass mode='train' to sample from the seeded P_art sequence
                out = model(batch_x, theta, mode=mode)

                # Cross Entropy Loss
                y_expanded = batch_y.unsqueeze(0).expand(model.K, -1)
                ce_loss = F.cross_entropy(out.reshape(-1, OUT_DIM), y_expanded.reshape(-1))
            else:
                # For Deterministic
                logits = model(batch_x, mode=mode)
                ce_loss = F.cross_entropy(logits, batch_y)
            
            loss = ce_loss
            reg_val = 0.0
            channel_shift_eval = 0.0
            channel_overfit_eval = 0.0
            model_complexity_eval = 0.0
            channel_overfit_kl_val = 0.0
            kl_ch_total_val = 0.0
            mixture_kl_val = 0.0
            component_expected_kl_val = 0.0
            k_hat_val = 0.0
            channel_penalty_val = 0.0
            k_hat_channel_penalty_val = 0.0

            if scenario_name == 'l2':
                l2_penalty = torch.zeros(1, device=ce_loss.device)
                for param in model.parameters():
                    l2_penalty = l2_penalty + torch.sum(param ** 2)
                reg = l2_penalty
                reg_val = reg.item()
                model_complexity_eval = reg_val
                if objective == 'regularization':
                    loss = ce_loss + l2_penalty
                elif objective == 'heuristic':
                    loss = ce_loss + (gamma_coeff * l2_penalty)
            # elif scenario_name == 'pac_bayes':
            #     # MC Sampling for bounds
            #     kl_mix, _ = compute_mixture_kl_and_channel_overfit(model, num_samples=mi_mc_samples)

            #     # Standard Regularization Formula
            #     reg = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (torch.clamp(kl_mix, min=0) + complexity_term))
            #     loss = ce_loss + (reg_coeff * reg)
            #     reg_val = reg.item()
            #     model_complexity_eval = reg.item()
            #     mixture_kl_val = kl_mix.item()
            elif scenario_name == 'proposed':
                if mode == 'perfect':
                    mixture_kl, _ = compute_mixture_kl_and_channel_overfit(model, num_samples=mi_mc_samples)
                    mixture_kl = torch.clamp(mixture_kl, min=0)
                    mixture_kl_val = mixture_kl.item()

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

                    channel_penalty_val = channel_penalty.item()

                    channel_shift = K_hat * channel_penalty
                    model_complexity = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (mixture_kl + complexity_term))
                    reg = channel_shift + model_complexity
                    channel_shift_eval = channel_shift.item()
                    model_complexity_eval = model_complexity.item()
                    k_hat_channel_penalty_val = channel_shift_eval
                    reg_val = reg.item()

                    if objective == 'bound':
                        loss = ce_loss + reg
                    elif objective == 'heuristic':
                        loss = ce_loss + (alpha_coeff * channel_shift + gamma_coeff * model_complexity)
                elif mode == 'train':
                    component_expected_kl = compute_component_expected_kl(model)
                    component_expected_kl_val = component_expected_kl.item()

                    mixture_kl, channel_overfit_kl = compute_mixture_kl_and_channel_overfit(
                        model,
                        num_samples=mi_mc_samples,
                    )
                    mixture_kl = torch.clamp(mixture_kl, min=0)
                    channel_overfit_kl = torch.clamp(channel_overfit_kl, min=0)

                    channel_shift = ce_loss.new_tensor(np.sqrt(2 * SIGMA_ART_SQ * kl_ch_total))
                    channel_overfit = torch.sqrt((2 * SIGMA_ART_SQ / m_artificial_channels) * channel_overfit_kl)
                    model_complexity = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (mixture_kl + complexity_term))
                    reg = channel_shift + channel_overfit + model_complexity
                    channel_shift_eval = channel_shift.item()
                    channel_overfit_eval = channel_overfit.item()
                    model_complexity_eval = model_complexity.item()
                    channel_overfit_kl_val = channel_overfit_kl.item()
                    mixture_kl_val = mixture_kl.item()
                    kl_ch_total_val = kl_ch_total
                    
                    if objective == 'bound':
                        loss = ce_loss + reg
                    elif objective == 'heuristic':
                        loss = ce_loss + (alpha_coeff * channel_shift + beta_coeff * channel_overfit + gamma_coeff * model_complexity)
                    reg_val = reg.item()

            if not torch.isfinite(loss):
                print(f"Stopping early: non-finite loss at epoch {epoch + 1}.")
                history['nonfinite_loss'] = True
                stop_training = True
                break

            loss.backward()
            optimizer.step()
            
            epoch_loss += ce_loss.item()
            epoch_reg += reg_val # Log the scaled applied bound
            epoch_bound_total += reg_val
            epoch_term1 += channel_shift_eval
            epoch_term2 += channel_overfit_eval
            epoch_term3 += model_complexity_eval
            epoch_channel_overfit_kl += channel_overfit_kl_val
            epoch_kl_ch_total += kl_ch_total_val
            epoch_mixture_kl += mixture_kl_val
            epoch_component_expected_kl += component_expected_kl_val
            epoch_k_hat += k_hat_val
            epoch_channel_penalty += channel_penalty_val
            epoch_k_hat_channel_penalty += k_hat_channel_penalty_val

        train_loss, train_acc = evaluate_model(model, loader, mode=mode)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['bound_total'].append(epoch_bound_total / len(loader))
        history['bound_term1'].append(epoch_term1 / len(loader))
        history['bound_term2'].append(epoch_term2 / len(loader))
        history['bound_term3'].append(epoch_term3 / len(loader))
        history['channel_overfit_kl'].append(epoch_channel_overfit_kl / len(loader))
        history['kl_ch_total'].append(epoch_kl_ch_total / len(loader))
        history['mixture_kl'].append(epoch_mixture_kl / len(loader))
        history['component_expected_kl'].append(epoch_component_expected_kl / len(loader))
        history['k_hat'].append(epoch_k_hat / len(loader))
        history['channel_penalty'].append(epoch_channel_penalty / len(loader))
        history['k_hat_channel_penalty'].append(epoch_k_hat_channel_penalty / len(loader))
        history['applied_regularization'].append(epoch_reg / len(loader))

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"CE Loss: {epoch_loss/len(loader):.4f} | "
                f"Reg Bound: {epoch_reg/len(loader):.4f} | "
                f"Bound T1: {epoch_term1/len(loader):.4f} | "
                f"Bound T2: {epoch_term2/len(loader):.4f} | "
                f"Bound T3: {epoch_term3/len(loader):.4f} | "
                f"Train Acc: {train_acc*100:.2f}%"
            )
        if scheduler is not None:
            scheduler.step()
        if stop_training:
            break

    if not history['train_loss']:
        history['train_loss'].append(float("nan"))
        history['train_acc'].append(0.0)
        history['bound_total'].append(float("nan"))
        history['bound_term1'].append(float("nan"))
        history['bound_term2'].append(float("nan"))
        history['bound_term3'].append(float("nan"))
        history['applied_regularization'].append(float("nan"))

    print(
        f"Epoch {EPOCHS}/{EPOCHS} | "
        f"CE Loss: {epoch_loss/len(loader):.4f} | "
        f"Reg Bound: {epoch_reg/len(loader):.4f} | "
        f"Bound T1: {epoch_term1/len(loader):.4f} | "
        f"Bound T2: {epoch_term2/len(loader):.4f} | "
        f"Bound T3: {epoch_term3/len(loader):.4f} | "
        f"Train Acc: {train_acc*100:.2f}%"
    )

    if use_cache:
        torch.save(model.state_dict(), weights_path)
        plot_training_metrics(history, scenario_name, mode, objective, run_dir)
        plot_bound_decomposition(history, scenario_name, mode, objective, run_dir)
        save_training_history(
            run_dir,
            scenario_name,
            mode,
            objective,
            history,
            lr,
            alpha_coeff,
            beta_coeff,
            gamma_coeff,
        )
    model.training_history = history
    model.converged = is_converged_history(history)
            
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
            total_expected_loss = 0.0
            total_correct = 0
            total = 0

            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)


                if getattr(model, "is_bnn", False):
                    # Monte Carlo over weight samples at test time (channel sampled inside forward)
                    total_mc = 0
                    batch_expected_loss = 0.0
                    batch_expectes_correct = 0.0

                    for _ in range(weight_mc_samples):
                        theta = model.sample_theta(1).squeeze(0)

                        out = model(batch_x, theta, mode='test')

                        # We calculate the loss for every model `K` independently,
                        # before any averaging of the logits.
                        y_expanded = batch_y.unsqueeze(0).expand(model.K, -1) # Shape: (K, B)
                        mc_loss = F.cross_entropy(out.reshape(-1, model.out_dim), y_expanded.reshape(-1))
                        batch_expected_loss += mc_loss.item()

                        preds = out.argmax(dim=-1) # Shape: (K, B)
                        mc_correct = (preds == y_expanded).sum().item() / model.K  # Average correct predictions across K components
                        batch_expectes_correct += mc_correct
                        total_mc += 1

                    # Average over MC samples
                    avg_batch_loss = batch_expected_loss / max(total_mc, 1)
                    avg_batch_correct = batch_expectes_correct / max(total_mc, 1)
                else:
                    # Deterministic model
                    logits = model(batch_x, mode='test')
                    loss = F.cross_entropy(logits, batch_y)
                    avg_batch_loss = loss.item()
                    preds = logits.argmax(dim=-1)
                    avg_batch_correct = (preds == batch_y).sum().item()

                total_expected_loss += avg_batch_loss * batch_y.size(0)
                total_correct += avg_batch_correct
                total += batch_y.size(0)

            loss_runs.append(total_expected_loss / max(total, 1))
            acc_runs.append(total_correct / max(total, 1))

    return float(np.mean(loss_runs)), float(np.mean(acc_runs))


def clipped_coeff(value, coeff_min, coeff_max, label, clip_log):
    clipped = float(np.clip(value, coeff_min, coeff_max))
    if clipped != float(value):
        clip_log.append({
            "coeff": label,
            "raw": float(value),
            "clipped": clipped,
        })
        print(f"Clipped {label}: raw={float(value):.6g}, clipped={clipped:.6g}")
    return clipped


def make_coeff_candidate(alpha, beta, gamma, source):
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "source": source,
    }


def build_fixed_coeff_candidates(include_alpha, compact=False):
    """
    Fixed, log-spaced regularization grid with a compact staged subset.
    This avoids calibrating coefficients from unstable untrained-model term magnitudes.
    """
    if compact:
        base_pairs = [
            (0.0, 0.03, 0.01),
            (0.0, 0.03, 0.03),
            (0.0, 0.1, 0.03),
            (0.0, 0.1, 0.1),
            (0.0, 0.3, 0.03),
            (0.0, 0.3, 0.1),
            (0.0, 1.0, 0.1),
        ]
        if include_alpha:
            base_pairs.extend((0.03, beta, gamma) for _, beta, gamma in base_pairs[:4])
        return [
            make_coeff_candidate(alpha, beta, gamma, "fixed_grid_compact")
            for alpha, beta, gamma in base_pairs
        ]

    candidates = {}
    for beta in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]:
        candidates[(0.0, round(beta, 12), 0.03)] = make_coeff_candidate(0.0, beta, 0.03, "fixed_grid_beta_axis")
    for gamma in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]:
        candidates[(0.0, 0.1, round(gamma, 12))] = make_coeff_candidate(0.0, 0.1, gamma, "fixed_grid_gamma_axis")
    for value in [0.01, 0.03, 0.1, 0.3, 1.0]:
        candidates[(0.0, round(value, 12), round(value, 12))] = make_coeff_candidate(0.0, value, value, "fixed_grid_diagonal")
    for beta, gamma in [(0.03, 0.1), (0.3, 0.01), (0.3, 0.3), (1.0, 0.03), (1.0, 0.3)]:
        candidates[(0.0, round(beta, 12), round(gamma, 12))] = make_coeff_candidate(0.0, beta, gamma, "fixed_grid_cross")

    if include_alpha:
        alpha_augmented = {}
        for _, beta, gamma in list(candidates.keys()):
            for alpha in [0.01, 0.03, 0.1]:
                alpha_augmented[(round(alpha, 12), beta, gamma)] = make_coeff_candidate(alpha, beta, gamma, "fixed_grid_alpha")
        candidates.update(alpha_augmented)

    return list(candidates.values())


def build_arch_candidates(hidden_dim_grid, n_u_sets_grid, m_artificial_channels_grid):
    candidates = []
    for hidden_dim in hidden_dim_grid:
        for n_u_sets in n_u_sets_grid:
            for m_artificial_channels in m_artificial_channels_grid:
                candidates.append({
                    "hidden_dim": int(hidden_dim),
                    "n_u_sets": int(n_u_sets),
                    "m_artificial_channels": int(m_artificial_channels),
                })
    return candidates


def architecture_key(record_or_spec):
    return (
        int(record_or_spec["hidden_dim"]),
        int(record_or_spec["n_u_sets"]),
        int(record_or_spec["m_artificial_channels"]),
    )


def run_proposed_candidate(
    stage,
    seed,
    train_loader,
    test_loader,
    n_trains,
    baseline_erm_acc,
    candidate,
    lr,
    hidden_dim,
    n_u_sets,
    m_artificial_channels,
):
    kl_ch_total = channel_kl_total(hidden_dim)
    prop_dir = get_run_dir(
        'proposed',
        'train',
        'heuristic',
        n_trains,
        seed,
        MI_MC_SAMPLES,
        LIPSCHITZ_METHOD_PERFECT,
        lr=lr,
        alpha_coeff=candidate["alpha"],
        beta_coeff=candidate["beta"],
        gamma_coeff=candidate["gamma"],
        hidden_dim=hidden_dim,
        n_u_sets=n_u_sets,
        m_artificial_channels=m_artificial_channels,
    )
    model_prop = train_scenario(
        'proposed',
        train_loader,
        n_trains,
        mode='train',
        objective='heuristic',
        lr=lr,
        alpha_coeff=candidate["alpha"],
        beta_coeff=candidate["beta"],
        gamma_coeff=candidate["gamma"],
        seed=seed,
        hidden_dim=hidden_dim,
        n_u_sets=n_u_sets,
        m_artificial_channels=m_artificial_channels,
        use_cache=False,
        run_dir=prop_dir,
    )
    loss_prop, acc_prop = evaluate_inference(
        model_prop,
        test_loader,
        repeats=EVAL_REPEATS,
        weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
    )
    inference = {"loss": loss_prop, "acc": acc_prop}
    update_run_history_inference(prop_dir, inference)

    return {
        "stage": stage,
        "seed": seed,
        "scenario": "proposed",
        "objective": "heuristic",
        "mode": "train",
        "lr": lr,
        "alpha": candidate["alpha"],
        "beta": candidate["beta"],
        "gamma": candidate["gamma"],
        "hidden_dim": hidden_dim,
        "n_u_sets": n_u_sets,
        "m_artificial_channels": m_artificial_channels,
        "kl_ch_total": kl_ch_total,
        "candidate": candidate,
        "loss": loss_prop,
        "acc": acc_prop,
        "baseline_erm_acc": baseline_erm_acc,
        "improvement": acc_prop - baseline_erm_acc,
        "converged": bool(getattr(model_prop, "converged", False)),
        "run_dir": prop_dir,
    }


def summarize_finalists(final_records, erm_records):
    finalist_groups = {}
    for record in final_records:
        key = (
            round(record["lr"], 12),
            round(record["alpha"], 12),
            round(record["beta"], 12),
            round(record["gamma"], 12),
            int(record["hidden_dim"]),
            int(record["n_u_sets"]),
            int(record["m_artificial_channels"]),
        )
        finalist_groups.setdefault(key, []).append(record)

    summaries = []
    for key, records in finalist_groups.items():
        mean_acc = float(np.mean([r["acc"] for r in records]))
        mean_erm = float(np.mean([r["baseline_erm_acc"] for r in records]))
        wins = int(sum(r["acc"] > r["baseline_erm_acc"] for r in records))
        summaries.append({
            "lr": key[0],
            "alpha": key[1],
            "beta": key[2],
            "gamma": key[3],
            "hidden_dim": key[4],
            "n_u_sets": key[5],
            "m_artificial_channels": key[6],
            "mean_acc": mean_acc,
            "mean_erm_acc": mean_erm,
            "mean_improvement": mean_acc - mean_erm,
            "wins": wins,
            "n_seeds": len(records),
            "success": mean_acc > mean_erm and wins > (len(records) / 2),
            "records": records,
        })

    summaries.sort(key=lambda r: (r["success"], r["mean_acc"], r["mean_improvement"]), reverse=True)
    best = summaries[0] if summaries else None
    winning_seed_records = [
        r for r in final_records
        if r.get("acc") is not None
        and r.get("baseline_erm_acc") is not None
        and r["acc"] > r["baseline_erm_acc"]
    ]
    winning_seed_records.sort(
        key=lambda r: (r["improvement"], r["acc"], r.get("converged", False)),
        reverse=True,
    )
    best_seed_param = winning_seed_records[0] if winning_seed_records else None
    return {
        "best_overall": best,
        "best_seed_param": best_seed_param,
        "winning_seed_params": winning_seed_records,
        "all_finalists": summaries,
        "erm_mean_acc": float(np.mean([r["acc"] for r in erm_records])) if erm_records else None,
        "success": bool(best and best["success"]),
    }

def free_memory(model):
    """Utility to explicitly delete models and flush MPS / CUDA cache."""
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    RUN_SWEEP = True
    EVAL_REPEATS = 10
    INFERENCE_WEIGHT_SAMPLES = 50

    if not RUN_SWEEP:
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

        # Scenario C: L2 Regularization + perfect channel
        model_l2_perfect = train_scenario('l2', train_loader, n_trains, mode='perfect', objective='heuristic', seed=SEED)
        loss_l2_perfect, acc_l2_perfect = evaluate_inference(
            model_l2_perfect,
            test_loader,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
        )

        # Scenario D: L2 Regularization + train channel
        model_l2 = train_scenario('l2', train_loader, n_trains, mode='train', objective='heuristic', seed=SEED)
        loss_l2, acc_l2 = evaluate_inference(
            model_l2,
            test_loader,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
        )

        # Scenario E: Proposed Bound + perfect channel
        model_prop_perfect = train_scenario('proposed', train_loader, n_trains, mode='perfect', objective='heuristic', seed=SEED)
        loss_prop_perfect, acc_prop_perfect = evaluate_inference(
            model_prop_perfect,
            test_loader,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
        )

        # Scenario F: Proposed Bound Regularization
        model_prop = train_scenario('proposed', train_loader, n_trains, mode='train', objective='heuristic', seed=SEED)
        loss_prop, acc_prop = evaluate_inference(
            model_prop,
            test_loader,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
        )

        print("\n" + "="*50)
        print("FINAL INFERENCE RESULTS (EVALUATED ON P_ch)")
        print("="*50)
        print(f"Standard ERM + Perfect Channel Loss/Acc: {loss_erm_perfect:.4f} / {acc_erm_perfect*100:.2f}%")
        print(f"Standard ERM Loss/Acc:                {loss_erm:.4f} / {acc_erm*100:.2f}%")
        print(f"L2 Reg + Perfect Channel Loss/Acc:    {loss_l2_perfect:.4f} / {acc_l2_perfect*100:.2f}%")
        print(f"L2 Reg + Train Channel Loss/Acc:      {loss_l2:.4f} / {acc_l2*100:.2f}%")
        print(f"Proposed Bound + Perfect Loss/Acc:    {loss_prop_perfect:.4f} / {acc_prop_perfect*100:.2f}%")
        print(f"Proposed Bound Reg Loss/Acc:          {loss_prop:.4f} / {acc_prop*100:.2f}%")
        print("="*50)
    else:
        # We use a JSON Lines (.jsonl) file to save incrementally without needing to load the array into RAM.
        results_jsonl_path = os.path.join(RESULTS_DIR, "sweep_results.jsonl")
        
        # Clear/Create the file at the start of the run
        with open(results_jsonl_path, "w", encoding="utf-8") as f:
            pass 

        # Sweep configuration
        SEEDS = [1, 2, 3, 4, 5]
        # HIDDEN_DIM_GRID = [16, 32, 64, 128]
        # N_U_SETS_GRID = [5, 10 ,15, 20]
        # M_ARTIFICIAL_CHANNELS_GRID = [10, 100, 500, 1000]
        HIDDEN_DIM_GRID = [64]
        N_U_SETS_GRID = [20]
        M_ARTIFICIAL_CHANNELS_GRID = [100]
        LR_GRID = [0.05, 0.01, 0.005, 0.003, 0.001, 0.0005]
        BETA_COEFF_GRID = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
        GAMMA_COEFF_GRID = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
        for seed in SEEDS:
            set_seed(seed)
            train_loader, test_loader, n_trains = get_dataloaders(
                N_SAMPLES,
                batch_size=BATCH_SIZE,
                seed=seed,
                noise=MOONS_NOISE,
            )

            for hidden_dim in HIDDEN_DIM_GRID:
                for n_u_sets in N_U_SETS_GRID:
                    for m_artificial_channels in M_ARTIFICIAL_CHANNELS_GRID:
                        for lr in LR_GRID:

                            # Baseline ERM (train channel) per seed
                            model_erm = train_scenario('erm', train_loader, n_trains, mode='train', objective='bound', hidden_dim=hidden_dim, n_u_sets=n_u_sets, m_artificial_channels=m_artificial_channels, lr=lr, seed=seed, use_cache=False)
                            loss_erm, acc_erm = evaluate_inference(
                                model_erm,
                                test_loader,
                                repeats=EVAL_REPEATS,
                                weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
                            )
                            erm_result = {
                                "seed": seed,
                                "scenario": "erm",
                                "objective": "bound",
                                "mode": "train",
                                "beta_coeff": None,
                                "gamma_coeff": None,
                                "hidden_dim": hidden_dim,
                                "n_u_sets": n_u_sets,
                                "m_artificial_channels": m_artificial_channels,
                                "lr": lr,
                                "loss": loss_erm,
                                "acc": acc_erm
                            }
                            
                            # Write incrementally
                            with open(results_jsonl_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(erm_result) + "\n")
                                
                            # Explicitly clear memory to avoid unified memory leaks on MPS/M-Series chips
                            free_memory(model_erm)



                            for beta_coeff in BETA_COEFF_GRID:
                                for gamma_coeff in GAMMA_COEFF_GRID:
                                    model_prop = train_scenario(
                                        'proposed',
                                        train_loader,
                                        n_trains,
                                        mode='train',
                                        objective='heuristic',
                                        hidden_dim=hidden_dim,
                                        n_u_sets=n_u_sets,
                                        m_artificial_channels=m_artificial_channels,
                                        lr=lr,
                                        alpha_coeff=0.0,
                                        beta_coeff=beta_coeff,
                                        gamma_coeff=gamma_coeff,
                                        seed=seed,
                                        use_cache=False
                                    )
                                    loss_prop, acc_prop = evaluate_inference(
                                        model_prop,
                                        test_loader,
                                        repeats=EVAL_REPEATS,
                                        weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
                                    )
                                    prop_result = {
                                        "seed": seed,
                                        "scenario": "proposed",
                                        "objective": "heuristic",
                                        "mode": "train",
                                        "hidden_dim": hidden_dim,
                                        "n_u_sets": n_u_sets,
                                        "m_artificial_channels": m_artificial_channels,
                                        "lr": lr,
                                        "beta_coeff": beta_coeff,
                                        "gamma_coeff": gamma_coeff,
                                        "loss": loss_prop,
                                        "acc": acc_prop
                                    }

                                    # Write incrementally
                                    with open(results_jsonl_path, "a", encoding="utf-8") as f:
                                        f.write(json.dumps(prop_result) + "\n")

        print(f"Sweep complete. Incremental results saved to {results_jsonl_path}")
        
        # --- End of computation. Post-process the file to calculate the summary. ---
        
        sweep_results = []
        with open(results_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                sweep_results.append(json.loads(line.strip()))

        # 1. Separate results into 'erm' and 'proposed'
        erm_results = [res for res in sweep_results if res['scenario'] == 'erm']
        proposed_results = [res for res in sweep_results if res['scenario'] == 'proposed']

        # 2. Find the best overall 'proposed' configuration
        best_proposed_overall = None
        if proposed_results:
            best_proposed_overall = max(proposed_results, key=lambda x: x['acc'])

        # 3. Build a lookup dictionary for ERM baselines for fast comparison
        # Key: (seed, hidden_dim, n_u_sets, m_artificial_channels, lr)
        # Value: erm_accuracy
        erm_baselines = {}
        for res in erm_results:
            key = (res['seed'], res['hidden_dim'], res['n_u_sets'], res['m_artificial_channels'], res['lr'])
            erm_baselines[key] = res['acc']

        # 4. Compare 'proposed' against its matching 'erm' baseline
        proposed_better_configs = []
        for prop in proposed_results:
            # Match using the shared hyperparameters
            key = (prop['seed'], prop['hidden_dim'], prop['n_u_sets'], prop['m_artificial_channels'], prop['lr'])
            
            # Get matching ERM accuracy (fallback to 0.0 if not found, though it should always be there)
            matching_erm_acc = erm_baselines.get(key, 0.0)
            
            # Check if proposed beats ERM
            if prop['acc'] > matching_erm_acc:
                prop_copy = prop.copy()
                prop_copy['erm_baseline_acc'] = matching_erm_acc
                prop_copy['accuracy_improvement'] = prop['acc'] - matching_erm_acc
                proposed_better_configs.append(prop_copy)

        # 5. Determine if it beats ERM and get the best outperforming configuration
        proposed_beats_erm_flag = len(proposed_better_configs) > 0
        best_beating_config = None

        if proposed_beats_erm_flag:
            # Find the config where proposed beat ERM by the largest margin
            best_beating_config = max(proposed_better_configs, key=lambda x: x['accuracy_improvement'])

        # 6. Construct the final summary dictionary
        summary_dict = {
            "best_proposed_overall": best_proposed_overall,
            "proposed_beats_erm_anywhere": proposed_beats_erm_flag,
            "total_configs_beating_erm": len(proposed_better_configs),
            "best_config_beating_erm_by_margin": best_beating_config
        }

        # 7. Save the summary to a JSON file
        summary_path = os.path.join(RESULTS_DIR, "sweep_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=4) # Indent=4 makes it highly readable

        print(f"Summary complete. JSON summary saved to {summary_path}")