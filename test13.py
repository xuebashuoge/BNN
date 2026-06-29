import os
import json
import hashlib
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import multiprocessing as mp
import concurrent.futures


# ERM using deterministic / proposed using new results Theorem 4 (given S), revise the sweeping, parallel

# ==========================================
# 0. OUTPUT PATHS
# ==========================================
RESULTS_DIR = os.path.join('results', 'test13')
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
LR_BASE = 0.003         # Adjusted base learning rate
LR_DECAY_STEP = 75     # StepLR decay period (epochs)
LR_DECAY_GAMMA = 0.8   # StepLR decay factor
PRIOR_LAMBDA = 1.0     # Variance of isotropic Gaussian prior
EPSILON = 0.025         # PAC-Bayes confidence parameter
SIGMA_SQ = 1.0         # Assumed sub-Gaussian parameter
SIGMA_ART_SQ = 1.0     # Assumed sub-Gaussian parameter for artificial-channel loss
ALPHA_COEFF = 0.0      # Weighting factor for the channel-shifting term in the objective
BETA_COEFF = 0.1       # Weighting factor for the channel-overfitting term in the objective
GAMMA_COEFF = 0.05     # Weighting factor for the standard PAC-Bayes term in the objective (optional ablation)
M_ARTIFICIAL_CHANNELS = 100  # m: size of the fixed artificial channel set U
MI_MC_SAMPLES = 100       # MC samples for mixture KL / channel-overfitting estimation
SEED = 8
LIPSCHITZ_METHOD_PERFECT = "grad"  # "grad" or "analytical"



# Channel Distributions (Inference/Test) - HARSH REALITY
P_OUTAGE_TE = 0.5

# Channel Distributions (Train) - DECEPTIVELY CLEAN
# ERM will become overconfident and build fragile decision boundaries.
P_OUTAGE_TR = P_OUTAGE_TE  

def kl_bernoulli(p_te, p_tr, eps=1e-6):
    """Closed form KL D(Bern(p_te) || Bern(p_tr)) where p is probability of outage."""
    p_te_c = np.clip(p_te, eps, 1.0 - eps)
    p_tr_c = np.clip(p_tr, eps, 1.0 - eps)
    # P(outage) = p, P(intact) = 1-p
    kl = p_te_c * np.log(p_te_c / p_tr_c) + (1 - p_te_c) * np.log((1 - p_te_c) / (1 - p_tr_c))
    return kl

def channel_kl_total(hidden_dim):
    """Total channel shift KL across independent hidden dimensions."""
    return hidden_dim * kl_bernoulli(P_OUTAGE_TE, P_OUTAGE_TR)

KL_CH_TOTAL = channel_kl_total(HIDDEN_DIM)
print(f"Default D(P_ch || P_art) = {KL_CH_TOTAL:.4f}")

def estimate_expected_channel_norm(hidden_dim, p_outage_te, norm_type="frobenius", device="cpu"):
    """Calculates the exact theoretical expected norm E[||M - I||] of a Bernoulli

    diagonal channel mask M, executed natively on the designated PyTorch device.
    """
    # 1. Input Validation
    if not (0.0 <= p_outage_te <= 1.0):
        raise ValueError(
            f"p_outage_te must be a valid probability in [0, 1], got {p_outage_te}"
        )

    n = int(hidden_dim)
    p_out = float(p_outage_te)
    norm_clean = norm_type.strip().lower()

    # Edge case: 0% outage means M is deterministically the Identity matrix
    if p_out == 0.0:
        return torch.tensor(0.0, device=device)

    # ---------------------------------------------------------
    # SPECTRAL NORM: E[ ||M - I||_2 ] = 1 - (1 - p_outage)^n
    # ---------------------------------------------------------
    if norm_clean in ["spectral", "l2", "2"]:
        # Probability that a single shifted diagonal entry is 0 is (1 - p_out)
        prob_entry_is_zero = torch.tensor(
            1.0 - p_out, device=device
        )

        prob_all_zero = torch.pow(prob_entry_is_zero, n)
        expected_spectral = 1.0 - prob_all_zero

        return expected_spectral

    # ---------------------------------------------------------
    # FROBENIUS NORM: Sum_{k=0}^{n} [ sqrt(k) * P(Binomial(n, p_out) == k) ]
    # ---------------------------------------------------------
    elif norm_clean in ["frobenius", "frob"]:
        # We use float64 to prevent exp(log_prob) underflow at extreme tails
        k_values = torch.arange(n + 1, device=device)

        binom = dist.Binomial(
            total_count=n,
            probs=torch.tensor(p_out, device=device),
        )

        # Compute PMF safely: exp( ln( P(X=k) ) )
        log_probs = binom.log_prob(k_values)
        pmf = torch.exp(log_probs)

        expected_frob = torch.sum(torch.sqrt(k_values) * pmf)

        return expected_frob

    else:
        raise ValueError(
            f"Unknown norm_type '{norm_type}'. Choose 'frobenius' or 'spectral'."
        )

CHANNEL_PENALTY_CACHE = {}

def get_channel_penalty(hidden_dim):
    if hidden_dim not in CHANNEL_PENALTY_CACHE:
        CHANNEL_PENALTY_CACHE[hidden_dim] = estimate_expected_channel_norm(
            hidden_dim,
            P_OUTAGE_TE,
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
base_data_path = os.environ.get('DATASET', './data')
data_path = os.path.join(base_data_path, 'two_moons')
os.makedirs(data_path, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_dataloaders(n_samples, batch_size=32, seed=0, noise=0.3):
    """Generates make_moons dataset and saves to DATASET path."""
    noise_tag = f"{noise:.4f}".replace(".", "p")
    dataset_file = os.path.join(data_path, f"moons_data_n{n_samples}_noise_{noise_tag}_seed_{seed}.pt")
    
    if not os.path.exists(dataset_file):
        print(f"Generating data and saving to {dataset_file}...")
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

        visualize_and_save_dataset(X, y, filename=os.path.join(data_path, f"moons_viz_seed_{seed}.png"), title="Original Moons Dataset")

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        torch.save((X_tensor, y_tensor), dataset_file)
    else:
        # Avoid printing on parallel workers to keep logs clean
        if mp.current_process().name == 'MainProcess':
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
    """Reduces dimensionality to 2D using PCA and saves a scatter plot."""
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_reduced[:, 0], 
        X_reduced[:, 1], 
        c=y, 
        cmap='viridis', 
        edgecolors='k', 
        alpha=0.7
    )
    plt.colorbar(scatter, label='Class Label')
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 4. ENSEMBLE BNN & STOCHASTIC CHANNEL LAYER
# ==========================================
class StochasticChannelLayer(nn.Module):
    def __init__(self, K, hid_dim, num_artificial_channels):
        super().__init__()
        self.K = K
        self.hid_dim = hid_dim
        self.num_artificial_channels = num_artificial_channels
        self.train_gen = torch.Generator(device='cpu')
        self.train_gen.manual_seed(1337)
        u_m = (torch.rand(K, num_artificial_channels, hid_dim, generator=self.train_gen) >= P_OUTAGE_TR).float()
        self.register_buffer("u_m", u_m)
        self.last_m = None

    def forward(self, x, mode='perfect'):
        B = x.shape[1]
        if mode == 'perfect':
            m = torch.ones_like(x, device=x.device)
        elif mode == 'train':
            idx = torch.randint(
                self.num_artificial_channels,
                (self.K, B),
                generator=self.train_gen,
                device='cpu',
            ).to(x.device)
            component_idx = torch.arange(self.K, device=x.device).unsqueeze(1)
            m = self.u_m.to(x.device)[component_idx, idx]
        elif mode == 'test':
            m = (torch.rand(self.K, B, self.hid_dim, device='cpu').to(x.device) >= P_OUTAGE_TE).float()
        else:
            raise ValueError(f"Invalid mode '{mode}'")

        self.last_m = m
        return x * m


class VectorizedBNNEnsemble(nn.Module):
    def __init__(self, K, in_dim, hid_dim, out_dim, num_artificial_channels):
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.is_bnn = True
        
        self.D1 = in_dim * hid_dim + hid_dim
        self.D2 = hid_dim * out_dim + out_dim
        self.D = self.D1 + self.D2
        
        self.mu = nn.Parameter(torch.randn(K, self.D) * 0.1)
        self.rho = nn.Parameter(torch.randn(K, self.D) * 0.1 - 3.0)
        self.channel_layer = StochasticChannelLayer(K, hid_dim, num_artificial_channels)

    def get_sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample_theta(self, num_samples=1):
        sigma = self.get_sigma()
        eps = torch.randn(num_samples, self.K, self.D, device='cpu').to(device)
        return self.mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

    def forward(self, x, theta, mode='perfect'):
        w1 = theta[:, :self.in_dim*self.hid_dim].view(self.K, self.in_dim, self.hid_dim)
        b1 = theta[:, self.in_dim*self.hid_dim:self.D1].view(self.K, self.hid_dim)
        w2 = theta[:, self.D1:self.D1+self.hid_dim*self.out_dim].view(self.K, self.hid_dim, self.out_dim)
        b2 = theta[:, self.D1+self.hid_dim*self.out_dim:].view(self.K, self.out_dim)
        
        B = x.shape[0]
        x_k = x.unsqueeze(0).expand(self.K, B, self.in_dim)
        
        h1 = torch.bmm(x_k, w1) + b1.unsqueeze(1)
        h1 = F.relu(h1)
        h1_ch = self.channel_layer(h1, mode=mode)
        out = torch.bmm(h1_ch, w2) + b2.unsqueeze(1)
        return out
    
    def get_sampled_weights(self, theta):
        w1 = theta[:, :self.in_dim*self.hid_dim].view(self.K, self.in_dim, self.hid_dim)
        b1 = theta[:, self.in_dim*self.hid_dim:self.D1].view(self.K, self.hid_dim)
        w2 = theta[:, self.D1:self.D1+self.hid_dim*self.out_dim].view(self.K, self.hid_dim, self.out_dim)
        b2 = theta[:, self.D1+self.hid_dim*self.out_dim:].view(self.K, self.out_dim)
        return [w1, b1, w2, b2]
    
    def compute_analytical_lipschitz(self, x, theta, mode='perfect'):
        w1, b1, w2, b2 = self.get_sampled_weights(theta)
        if w2.is_mps:
            w2_sn = torch.linalg.matrix_norm(w2.cpu(), ord=2).to(w2.device)
        else:
            w2_sn = torch.linalg.matrix_norm(w2, ord=2)
        
        with torch.no_grad():
            B = x.shape[0]
            x_k = x.unsqueeze(0).expand(self.K, B, self.in_dim)
            h1 = torch.bmm(x_k, w1) + b1.unsqueeze(1)
            h1 = F.relu(h1)
            h1_ch = self.channel_layer(h1, mode=mode)
            m = self.channel_layer.last_m
        
        x_norm = torch.norm(x, p=2, dim=1).max()
        h1_ch_norm = torch.norm(h1_ch, p=2, dim=2).max(dim=1)[0]
        m_max = torch.abs(m).max(dim=2)[0].max(dim=1)[0] if m is not None else torch.ones(self.K, device=device)
        
        ce_lip = 1.414 
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
    var = sigma**2 + 1e-8
    log_scale = torch.log(var) * 0.5 + 0.5 * np.log(2 * np.pi)
    diff = (theta - mu.unsqueeze(0))**2 / (2 * var.unsqueeze(0))
    return -torch.sum(log_scale.unsqueeze(0) + diff, dim=-1)

def log_prior(theta, prior_lambda):
    var = prior_lambda
    log_scale = 0.5 * np.log(var) + 0.5 * np.log(2 * np.pi)
    diff = theta**2 / (2 * var)
    return -torch.sum(log_scale + diff, dim=-1)

def compute_component_expected_kl(model):
    mu = model.mu
    sigma = model.get_sigma()
    var = sigma**2
    kl_k = 0.5 * torch.sum(var/PRIOR_LAMBDA + mu**2/PRIOR_LAMBDA - 1 - torch.log(var/PRIOR_LAMBDA), dim=-1)
    return kl_k.mean()

def compute_mixture_kl_and_channel_overfit(model, num_samples=3):
    theta = model.sample_theta(num_samples)
    mu = model.mu
    sigma = model.get_sigma()
    
    log_q_k = log_gaussian(theta, mu, sigma)
    
    theta_exp = theta.unsqueeze(2)
    mu_exp = mu.unsqueeze(0).unsqueeze(0)
    sigma_exp = sigma.unsqueeze(0).unsqueeze(0) 
    
    var_exp = sigma_exp**2 + 1e-8
    log_scale = torch.log(var_exp) * 0.5 + 0.5 * np.log(2 * np.pi)
    diff = (theta_exp - mu_exp)**2 / (2 * var_exp)

    log_q_all = -torch.sum(log_scale + diff, dim=-1)
    log_p_mix = torch.logsumexp(log_q_all, dim=2) - np.log(model.K)
    
    channel_overfit = (log_q_k - log_p_mix).mean()
    log_q_prior = log_prior(theta, PRIOR_LAMBDA)
    kl_mix = (log_p_mix - log_q_prior).mean()
    
    return kl_mix, channel_overfit

def evaluate_model(model, loader, mode='train'):
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
                y_expanded = batch_y.unsqueeze(0).expand(model.K, -1)
                loss = F.cross_entropy(out.reshape(-1, model.out_dim), y_expanded.reshape(-1))
                preds = out.argmax(dim=-1)
                total_correct += (preds == y_expanded).sum().item() / model.K 
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

def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

def build_param_config(scenario_name, mode, objective, n_samples, seed, mi_mc_samples, lipschitz_method_perfect, batch_size=None, moon_noise=None, p_outage_te=None, p_outage_tr=None, epochs=None, lr=None, lr_decay_step=None, lr_decay_gamma=None, alpha_coeff=None, beta_coeff=None, gamma_coeff=None, hidden_dim=None, n_u_sets=None, m_artificial_channels=None):
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    moon_noise = MOONS_NOISE if moon_noise is None else moon_noise
    p_outage_te = P_OUTAGE_TE if p_outage_te is None else p_outage_te
    p_outage_tr = P_OUTAGE_TR if p_outage_tr is None else p_outage_tr
    epochs = EPOCHS if epochs is None else epochs
    lr = LR_BASE if lr is None else lr
    lr_decay_step = LR_DECAY_STEP if lr_decay_step is None else lr_decay_step
    lr_decay_gamma = LR_DECAY_GAMMA if lr_decay_gamma is None else lr_decay_gamma
    alpha_coeff = ALPHA_COEFF if alpha_coeff is None else alpha_coeff
    beta_coeff = BETA_COEFF if beta_coeff is None else beta_coeff
    gamma_coeff = GAMMA_COEFF if gamma_coeff is None else gamma_coeff
    hidden_dim = HIDDEN_DIM if hidden_dim is None else hidden_dim
    n_u_sets = N_U_SETS if n_u_sets is None else n_u_sets
    m_artificial_channels = M_ARTIFICIAL_CHANNELS if m_artificial_channels is None else m_artificial_channels
    return {
        "scenario_name": scenario_name, "mode": mode,
        "objective": "N/A" if scenario_name == 'erm' else objective,
        "n_samples": n_samples, "seed": seed,
        "batch_size": batch_size, "moon_noise": moon_noise,
        "training": {
            "epochs": epochs,
            "lr": lr, "lr_decay_step": lr_decay_step, "lr_decay_gamma": lr_decay_gamma,
            "alpha_coeff": alpha_coeff if scenario_name != 'erm' else "N/A",
            "beta_coeff": beta_coeff if scenario_name != 'erm' else "N/A",
            "gamma_coeff": gamma_coeff if scenario_name != 'erm' else "N/A",
            "hidden_dim": hidden_dim, 
            "n_u_sets": n_u_sets if scenario_name != 'erm' else "N/A",
            "m_artificial_channels": m_artificial_channels,
        }
    }

def get_run_dir(scenario_name, mode, objective, n_samples, seed, mi_mc_samples, lipschitz_method_perfect, batch_size=None, moon_noise=None, p_outage_te=None, p_outage_tr=None, epochs=None, lr=None, lr_decay_step=None, lr_decay_gamma=None, alpha_coeff=None, beta_coeff=None, gamma_coeff=None, hidden_dim=None, n_u_sets=None, m_artificial_channels=None):
    param_cfg = build_param_config(scenario_name, mode, objective, n_samples, seed, mi_mc_samples, lipschitz_method_perfect, batch_size=batch_size, moon_noise=moon_noise, p_outage_te=p_outage_te, p_outage_tr=p_outage_tr, epochs=epochs, lr=lr, lr_decay_step=lr_decay_step, lr_decay_gamma=lr_decay_gamma, alpha_coeff=alpha_coeff, beta_coeff=beta_coeff, gamma_coeff=gamma_coeff, hidden_dim=hidden_dim, n_u_sets=n_u_sets, m_artificial_channels=m_artificial_channels)
    run_id = config_hash(param_cfg)
    label = scenario_label(scenario_name, mode, objective)
    run_dir = os.path.join(RESULTS_DIR, label, f"param_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
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
    batch_size,
    moon_noise,
    n_samples,
    epochs,
    history,
    lr,
    lr_decay_step,
    lr_decay_gamma,
    alpha_coeff,
    beta_coeff,
    gamma_coeff,
    inference=None,
):
    payload = {
        "scenario": scenario_name,
        "mode": mode,
        "objective": objective,
        "batch_size": batch_size,
        "moon_noise": moon_noise,
        "n_samples": n_samples,
        "epochs": epochs,
        "lr": lr,
        "lr_decay_step": lr_decay_step,
        "lr_decay_gamma": lr_decay_gamma,
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

def train_scenario(scenario_name, loader, n_samples, mode='perfect', objective='bound', batch_size=None, moon_noise=None, p_outage_te=None, p_outage_tr=None, epochs=None, lr=None, lr_decay_step=None, lr_decay_gamma=None, alpha_coeff=None, beta_coeff=None, gamma_coeff=None, mi_mc_samples=None, seed=None, lipschitz_method_perfect=None, hidden_dim=None, n_u_sets=None, m_artificial_channels=None, use_cache=True, run_dir=None, verbose=False):
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    moon_noise = MOONS_NOISE if moon_noise is None else moon_noise
    p_outage_te = P_OUTAGE_TE if p_outage_te is None else p_outage_te
    p_outage_tr = P_OUTAGE_TR if p_outage_tr is None else p_outage_tr
    epochs = EPOCHS if epochs is None else epochs
    lr = LR_BASE if lr is None else lr
    lr_decay_step = LR_DECAY_STEP if lr_decay_step is None else lr_decay_step
    lr_decay_gamma = LR_DECAY_GAMMA if lr_decay_gamma is None else lr_decay_gamma
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
    if lr_decay_step and lr_decay_step > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_decay_step,
            gamma=lr_decay_gamma,
        )

    mi_mc_samples = MI_MC_SAMPLES if mi_mc_samples is None else mi_mc_samples
    lipschitz_method_perfect = LIPSCHITZ_METHOD_PERFECT if lipschitz_method_perfect is None else lipschitz_method_perfect
    if lipschitz_method_perfect not in {"grad", "analytical"}:
        raise ValueError("lipschitz_method_perfect must be 'grad' or 'analytical'.")
    if run_dir is None and use_cache:
        run_dir = get_run_dir(
            scenario_name, mode, objective,
            n_samples, seed, mi_mc_samples, lipschitz_method_perfect, batch_size=batch_size, moon_noise=moon_noise, p_outage_te=p_outage_te, p_outage_tr=p_outage_tr, epochs=epochs,
            lr=lr, lr_decay_step=lr_decay_step, lr_decay_gamma=lr_decay_gamma, alpha_coeff=alpha_coeff, beta_coeff=beta_coeff, gamma_coeff=gamma_coeff,
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

    for epoch in range(epochs):
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

        if verbose:
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

    if verbose:
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
            batch_size,
            moon_noise,
            n_samples,
            epochs,
            history,
            lr,
            lr_decay_step,
            lr_decay_gamma,
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
def evaluate_inference(model, loader, seed, repeats=1, weight_mc_samples=1):
    model.eval()
    loss_runs = []
    acc_runs = []

    set_seed(seed)

    with torch.no_grad():
        for _ in range(repeats):
            total_expected_loss = 0.0
            total_correct = 0
            total = 0

            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                if getattr(model, "is_bnn", False):
                    total_mc = 0
                    batch_expected_loss = 0.0
                    batch_expectes_correct = 0.0

                    for _ in range(weight_mc_samples):
                        theta = model.sample_theta(1).squeeze(0)
                        out = model(batch_x, theta, mode='test')

                        y_expanded = batch_y.unsqueeze(0).expand(model.K, -1) 
                        mc_loss = F.cross_entropy(out.reshape(-1, model.out_dim), y_expanded.reshape(-1))
                        batch_expected_loss += mc_loss.item()

                        preds = out.argmax(dim=-1) 
                        mc_correct = (preds == y_expanded).sum().item() / model.K 
                        batch_expectes_correct += mc_correct
                        total_mc += 1

                    avg_batch_loss = batch_expected_loss / max(total_mc, 1)
                    avg_batch_correct = batch_expectes_correct / max(total_mc, 1)
                else:
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

def free_memory(model):
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================
# PARALLEL WORKER FUNCTION
# ==========================================
def run_sweep_task(task_config, repeats=1, weight_mc_samples=1):
    """
    Executes a single model configuration safely in a parallel process.
    """
    # Prevent PyTorch from using multiple threads per parallel worker (prevents thrashing)
    torch.set_num_threads(1)
    
    seed = task_config['seed']
    scenario = task_config['scenario']
    hidden_dim = task_config['hidden_dim']
    batch_size = task_config['batch_size']
    moon_noise = task_config['moon_noise']
    n_samples = task_config['n_samples']
    epochs = task_config['epochs']
    m_artificial_channels = task_config['m_artificial_channels']
    lr = task_config['lr']
    lr_decay_step = task_config['lr_decay_step']
    lr_decay_gamma = task_config['lr_decay_gamma']
    use_cache = task_config['use_cache']
    verbose = task_config['verbose']
    p_outage_te = task_config.get('p_outage_te', P_OUTAGE_TE)
    p_outage_tr = task_config.get('p_outage_tr', P_OUTAGE_TR)

    set_seed(seed)
    
    # Load safe pre-generated datasets from disk
    train_loader, test_loader, n_trains = get_dataloaders(
        n_samples,
        batch_size=batch_size,
        seed=seed,
        noise=moon_noise,
    )

    if scenario == 'erm':
        model = train_scenario(
            'erm', train_loader, n_trains, mode='train', objective='bound', 
            hidden_dim=hidden_dim, epochs=epochs, batch_size=batch_size, moon_noise=moon_noise, p_outage_te=p_outage_te, p_outage_tr=p_outage_tr,
            m_artificial_channels=m_artificial_channels, lr=lr, lr_decay_step=lr_decay_step, lr_decay_gamma=lr_decay_gamma, seed=seed, use_cache=use_cache, verbose=verbose
        )
        loss, acc = evaluate_inference(model, test_loader, seed=seed, repeats=repeats, weight_mc_samples=weight_mc_samples)
        free_memory(model)
        
        return {
            "seed": seed, "scenario": "erm", "objective": "bound", "mode": "train",
            "beta_coeff": None, "gamma_coeff": None, "hidden_dim": hidden_dim, "batch_size": batch_size, "moon_noise": moon_noise, "n_samples": n_samples, "epochs": epochs,
            "n_u_sets": None, "m_artificial_channels": m_artificial_channels,
            "lr": lr, "lr_decay_step": lr_decay_step, "lr_decay_gamma": lr_decay_gamma, "loss": loss, "acc": acc
        }
        
    elif scenario == 'proposed':
        beta_coeff = task_config['beta_coeff']
        gamma_coeff = task_config['gamma_coeff']
        n_u_sets = task_config['n_u_sets']
        
        model = train_scenario(
            'proposed', train_loader, n_trains, mode='train', objective='heuristic',
            hidden_dim=hidden_dim, batch_size=batch_size, moon_noise=moon_noise, p_outage_te=p_outage_te, p_outage_tr=p_outage_tr, epochs=epochs,
            n_u_sets=n_u_sets, m_artificial_channels=m_artificial_channels,
            lr=lr, lr_decay_step=lr_decay_step, lr_decay_gamma=lr_decay_gamma, alpha_coeff=0.0, beta_coeff=beta_coeff, gamma_coeff=gamma_coeff,
            seed=seed, use_cache=use_cache, verbose=verbose
        )
        loss, acc = evaluate_inference(model, test_loader, seed=seed, repeats=repeats, weight_mc_samples=weight_mc_samples)
        free_memory(model)
        
        return {
            "seed": seed, "scenario": "proposed", "objective": "heuristic", "mode": "train",
            "hidden_dim": hidden_dim, "batch_size": batch_size, "moon_noise": moon_noise, "p_outage_te": p_outage_te, "p_outage_tr": p_outage_tr, "n_samples": n_samples, "epochs": epochs,
            "n_u_sets": n_u_sets, "m_artificial_channels": m_artificial_channels,
            "lr": lr, "lr_decay_step": lr_decay_step, "lr_decay_gamma": lr_decay_gamma, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
            "loss": loss, "acc": acc
        }


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    RUN_SWEEP = False

    EVAL_REPEATS = 10
    INFERENCE_WEIGHT_SAMPLES = 50

    if not RUN_SWEEP:
        set_seed(SEED)
        train_loader, test_loader, n_trains = get_dataloaders(
            N_SAMPLES,
            batch_size=BATCH_SIZE,
            seed=SEED,
            noise=MOONS_NOISE,
        )
        # Scenario A: Standard ERM + perfect channel
        model_erm_perfect = train_scenario('erm', train_loader, n_trains, mode='perfect', objective='bound', seed=SEED, use_cache=True, verbose=True)
        loss_erm_perfect, acc_erm_perfect = evaluate_inference(
            model_erm_perfect,
            test_loader,
            seed=SEED,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES,
        )

        # Scenario B: Standard ERM + train channel (overfitting to P_art)
        model_erm = train_scenario('erm', train_loader, n_trains, mode='train', objective='bound', seed=SEED, use_cache=True, verbose=True)
        loss_erm, acc_erm = evaluate_inference(
            model_erm,
            test_loader,
            seed=SEED,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES,
        )

        # Scenario C: L2 Regularization + perfect channel
        model_l2_perfect = train_scenario('l2', train_loader, n_trains, mode='perfect', objective='heuristic', seed=SEED, use_cache=True, verbose=True)
        loss_l2_perfect, acc_l2_perfect = evaluate_inference(
            model_l2_perfect,
            test_loader,
            seed=SEED,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES,
        )

        # Scenario D: L2 Regularization + train channel
        model_l2 = train_scenario('l2', train_loader, n_trains, mode='train', objective='heuristic', seed=SEED, use_cache=True, verbose=True)
        loss_l2, acc_l2 = evaluate_inference(
            model_l2,
            test_loader,
            seed=SEED,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES,
        )

        # Scenario E: Proposed Bound + perfect channel
        model_prop_perfect = train_scenario('proposed', train_loader, n_trains, mode='perfect', objective='heuristic', seed=SEED, use_cache=True, verbose=True)
        loss_prop_perfect, acc_prop_perfect = evaluate_inference(
            model_prop_perfect,
            test_loader,
            seed=SEED,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES,
        )

        # Scenario F: Proposed Bound Regularization
        model_prop = train_scenario('proposed', train_loader, n_trains, mode='train', objective='heuristic', seed=SEED, use_cache=True, verbose=True)
        loss_prop, acc_prop = evaluate_inference(
            model_prop,
            test_loader,
            seed=SEED,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES,
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
        # Ensure safe context for Linux multiprocessing with PyTorch
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        results_jsonl_path = os.path.join(RESULTS_DIR, "sweep_results.jsonl")
        with open(results_jsonl_path, "w", encoding="utf-8") as f:
            pass 

        # 1. Set a master seed for reproducibility
        MASTER_SEED = 42
        random.seed(MASTER_SEED)

        # 2. Generate a list of 10 unique random seeds
        num_seeds = 10
        # Seeds in Python/NumPy are typically unsigned 32-bit integers (0 to 2**32 - 1)
        SEEDS = [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]
        # SEEDS = [2010133918]
        HIDDEN_DIM_GRID = [64]
        N_U_SETS_GRID = [10]
        M_ARTIFICIAL_CHANNELS_GRID = [100]
        LR_GRID = [0.001, 0.003, 0.005, 0.007, 0.01]
        LR_DECAY_STEP_GRID = [25, 50, 75]
        LR_DECAY_GAMMA_GRID = [0.1, 0.5, 0.8]
        BETA_COEFF_GRID = [0.5, 0.1, 0.05, 0.01]
        GAMMA_COEFF_GRID = [0.5, 0.1, 0.05, 0.01, 0.005]
        EPOCHS_GRID = [150]
        use_cache = False
        verbose = False

        # PRE-GENERATE DATASETS SEQUENTIALLY
        # This prevents multiple worker processes from trying to create and write the .pt file at the same time.
        print("Pre-generating dataset .pt files to prevent race conditions...")
        for seed in SEEDS:
            get_dataloaders(N_SAMPLES, batch_size=BATCH_SIZE, seed=seed, noise=MOONS_NOISE)

        # BUILD TASK LIST
        print("Building task queue...")
        tasks = []
        for seed in SEEDS:
            for hidden_dim in HIDDEN_DIM_GRID:
                for n_u_sets in N_U_SETS_GRID:
                    for m_artificial_channels in M_ARTIFICIAL_CHANNELS_GRID:
                        for epochs in EPOCHS_GRID:
                            for lr in LR_GRID:
                                for lr_decay_step in LR_DECAY_STEP_GRID:
                                    for lr_decay_gamma in LR_DECAY_GAMMA_GRID:
                                        # Append ERM Task
                                        tasks.append({
                                            "scenario": "erm", "seed": seed, "hidden_dim": hidden_dim, "batch_size": BATCH_SIZE, "moon_noise": MOONS_NOISE, "n_samples": N_SAMPLES, "epochs": epochs,
                                            "m_artificial_channels": m_artificial_channels, "lr": lr,
                                            "lr_decay_step": lr_decay_step, "lr_decay_gamma": lr_decay_gamma,
                                            "use_cache": use_cache, "verbose": verbose,
                                        })
                                        
                                        # Append Proposed Tasks
                                        for beta_coeff in BETA_COEFF_GRID:
                                            for gamma_coeff in GAMMA_COEFF_GRID:
                                                tasks.append({
                                                    "scenario": "proposed", "seed": seed, "hidden_dim": hidden_dim, "batch_size": BATCH_SIZE, "moon_noise": MOONS_NOISE, "n_samples": N_SAMPLES, "epochs": epochs,
                                                    "n_u_sets": n_u_sets, "m_artificial_channels": m_artificial_channels, 
                                                    "lr": lr, "lr_decay_step": lr_decay_step, "lr_decay_gamma": lr_decay_gamma, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
                                                    "use_cache": use_cache, "verbose": verbose,
                                                })

        print(f"Total tasks generated: {len(tasks)}")

        # EXECUTE IN PARALLEL
        # Determine safe number of workers (leave 1 core free for OS)
        MAX_WORKERS = max(1, os.cpu_count() - 1)
        # MAX_WORKERS = 2
        # If running on a powerful server, you might cap this at 32 so you don't overwhelm I/O
        if MAX_WORKERS > 64: 
            MAX_WORKERS = 64
            
        print(f"Starting parallel execution with {MAX_WORKERS} workers...")

        # We use ProcessPoolExecutor. `as_completed` allows us to safely write to the file 
        # from this main thread as results return, avoiding file locking issues.
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(run_sweep_task, task, EVAL_REPEATS, INFERENCE_WEIGHT_SAMPLES): task for task in tasks}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                completed += 1
                try:
                    result = future.result()
                    with open(results_jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result) + "\n")
                        
                    if completed % 50 == 0:
                        print(f"Progress: {completed}/{len(tasks)} tasks completed.")
                except Exception as exc:
                    print(f"A task generated an exception: {exc}")

        print(f"Sweep complete. Incremental results saved to {results_jsonl_path}")
        
        # --- Post-process the file to calculate the summary ---
        sweep_results = []
        with open(results_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                sweep_results.append(json.loads(line.strip()))

        erm_results = [res for res in sweep_results if res['scenario'] == 'erm']
        proposed_results = [res for res in sweep_results if res['scenario'] == 'proposed']

        best_proposed_overall = None
        if proposed_results:
            best_proposed_overall = max(proposed_results, key=lambda x: x['acc'])

        erm_baselines = {}
        for res in erm_results:
            key = (res['seed'], res['hidden_dim'], res['batch_size'], res['moon_noise'], res['epochs'], res['n_samples'], res['m_artificial_channels'], res['lr'], res['lr_decay_step'], res['lr_decay_gamma'])
            erm_baselines[key] = res['acc']

        proposed_better_configs = []
        for prop in proposed_results:
            key = (prop['seed'], prop['hidden_dim'], prop['batch_size'], prop['moon_noise'], prop['epochs'], prop['n_samples'], prop['m_artificial_channels'], prop['lr'], prop['lr_decay_step'], prop['lr_decay_gamma'])
            matching_erm_acc = erm_baselines.get(key, 0.0)
            
            if prop['acc'] > matching_erm_acc:
                prop_copy = prop.copy()
                prop_copy['erm_baseline_acc'] = matching_erm_acc
                prop_copy['accuracy_improvement'] = prop['acc'] - matching_erm_acc
                proposed_better_configs.append(prop_copy)

        proposed_beats_erm_flag = len(proposed_better_configs) > 0
        best_beating_config = None

        if proposed_beats_erm_flag:
            best_beating_config = max(proposed_better_configs, key=lambda x: x['accuracy_improvement'])

        summary_dict = {
            "best_proposed_overall": best_proposed_overall,
            "proposed_beats_erm_anywhere": proposed_beats_erm_flag,
            "total_configs_beating_erm": len(proposed_better_configs),
            "best_config_beating_erm_by_margin": best_beating_config
        }

        summary_path = os.path.join(RESULTS_DIR, "sweep_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=4)

        print(f"Summary complete. JSON summary saved to {summary_path}")