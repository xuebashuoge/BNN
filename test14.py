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

# ==========================================
# 0. OUTPUT PATHS
# ==========================================
RESULTS_DIR = os.path.join('results', 'test14')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 1. HARDWARE & PATH CONFIGURATION (CRITICAL)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 2. HYPERPARAMETERS & CHANNEL DISTRIBUTIONS
# ==========================================
N_SAMPLES = 1000       # Total samples in the dataset
N_U_SETS = 10          # Number of independently sampled U sets / posterior components
HIDDEN_DIM = 64       # Capacity of the network
IN_DIM = 2
OUT_DIM = 2
MOONS_NOISE = 0.3      # make_moons noise level
BATCH_SIZE = 64        # Reduced batch size for noisier gradients
EPOCHS = 150           # Epochs to ensure ERM fully memorizes
LR_BASE = 0.005         # Adjusted base learning rate
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

SEED = 2846182436
P_OUTAGE_TE = 0.5  # Fixed default test channel outage probability
P_OUTAGE_TR = 0.5  # Default train channel outage probability (will be swept in main)

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

def build_param_config(scenario_name, mode, objective, n_samples, seed, mi_mc_samples, lipschitz_method_perfect, batch_size=None, moon_noise=None, epochs=None, lr=None, lr_decay_step=None, lr_decay_gamma=None, alpha_coeff=None, beta_coeff=None, gamma_coeff=None, hidden_dim=None, n_u_sets=None, m_artificial_channels=None, is_bnn=False):
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    moon_noise = MOONS_NOISE if moon_noise is None else moon_noise
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
        "is_bnn": is_bnn,
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

def get_run_dir(scenario_name, mode, objective, n_samples, seed, mi_mc_samples, lipschitz_method_perfect, batch_size=None, moon_noise=None, epochs=None, lr=None, lr_decay_step=None, lr_decay_gamma=None, alpha_coeff=None, beta_coeff=None, gamma_coeff=None, hidden_dim=None, n_u_sets=None, m_artificial_channels=None, is_bnn=False):
    param_cfg = build_param_config(scenario_name, mode, objective, n_samples, seed, mi_mc_samples, lipschitz_method_perfect, batch_size=batch_size, moon_noise=moon_noise, epochs=epochs, lr=lr, lr_decay_step=lr_decay_step, lr_decay_gamma=lr_decay_gamma, alpha_coeff=alpha_coeff, beta_coeff=beta_coeff, gamma_coeff=gamma_coeff, hidden_dim=hidden_dim, n_u_sets=n_u_sets, m_artificial_channels=m_artificial_channels, is_bnn=is_bnn)
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
            n_samples, seed, mi_mc_samples, lipschitz_method_perfect, batch_size=batch_size, moon_noise=moon_noise, epochs=epochs,
            lr=lr, lr_decay_step=lr_decay_step, lr_decay_gamma=lr_decay_gamma, alpha_coeff=alpha_coeff, beta_coeff=beta_coeff, gamma_coeff=gamma_coeff,
            hidden_dim=hidden_dim, n_u_sets=n_u_sets, m_artificial_channels=m_artificial_channels,
            is_bnn=model.is_bnn,
        )
    else:
        run_dir = run_dir or "temp_run"

    weights_path = os.path.join(run_dir, "weights.pth")

    if use_cache and os.path.exists(weights_path):
        print(f"Loading existing weights: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        with open(os.path.join(run_dir, "history.json"), "r", encoding="utf-8") as f:
            history = json.load(f).get("history", {})
        return model, history

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
            bound_val = 0.0
            channel_shift_eval = 0.0
            channel_overfit_eval = 0.0
            model_complexity_eval = 0.0
            channel_overfit_kl_val = 0.0
            kl_ch_total_val = 0.0
            mixture_kl_val = 0.0
            component_expected_kl_val = 0.0
            k_hat_val = 0.0
            channel_penalty_val = 0.0

            # calculate the exact derived bound for calculation
            if model.is_bnn:
                mixture_kl, channel_overfit_kl = compute_mixture_kl_and_channel_overfit(
                    model,
                    num_samples=mi_mc_samples,
                )
            else:
                mixture_kl = torch.tensor(0.0, device=ce_loss.device)
                channel_overfit_kl = torch.tensor(0.0, device=ce_loss.device)

            mixture_kl = torch.clamp(mixture_kl, min=0)
            channel_overfit_kl = torch.clamp(channel_overfit_kl, min=0)
            channel_overfit_kl_val = channel_overfit_kl.item()
            mixture_kl_val = mixture_kl.item()
            if mode == 'perfect' and model.is_bnn:
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

                
                channel_shift = K_hat * channel_penalty

                model_complexity = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (mixture_kl + complexity_term))

                channel_shift_eval = channel_shift.item()
                model_complexity_eval = model_complexity.item()
                # the expected norm distance between channel matrix and identity matrix
                channel_penalty_val = channel_penalty.item()

                bound = channel_shift + model_complexity
                bound_val = bound.item()

            elif mode == 'train':
                channel_shift = ce_loss.new_tensor(np.sqrt(2 * SIGMA_ART_SQ * kl_ch_total))

                channel_overfit = torch.sqrt((2 * SIGMA_ART_SQ / m_artificial_channels) * channel_overfit_kl)

                model_complexity = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (mixture_kl + complexity_term))

                channel_shift_eval = channel_shift.item()
                channel_overfit_eval = channel_overfit.item()
                model_complexity_eval = model_complexity.item()
                kl_ch_total_val = kl_ch_total

                bound = channel_shift + channel_overfit + model_complexity
                bound_val = bound.item()

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
            elif scenario_name == 'proposed':
                if mode == 'perfect':
                    reg = bound
                    reg_val = reg.item()

                    if objective == 'bound':
                        loss = ce_loss + reg
                    elif objective == 'heuristic':
                        loss = ce_loss + (alpha_coeff * channel_shift + gamma_coeff * model_complexity)
                elif mode == 'train':
                    component_expected_kl = compute_component_expected_kl(model)
                    component_expected_kl_val = component_expected_kl.item()

                    
                    reg = bound
                    reg_val = reg.item()
                    
                    if objective == 'bound':
                        loss = ce_loss + reg
                    elif objective == 'heuristic':
                        loss = ce_loss + (alpha_coeff * channel_shift + beta_coeff * channel_overfit + gamma_coeff * model_complexity)

            if not torch.isfinite(loss):
                print(f"Stopping early: non-finite loss at epoch {epoch + 1}.")
                history['nonfinite_loss'] = True
                stop_training = True
                break

            loss.backward()
            optimizer.step()
            
            epoch_loss += ce_loss.item()
            epoch_reg += reg_val # Log the scaled applied bound
            epoch_bound_total += bound_val
            epoch_term1 += channel_shift_eval
            epoch_term2 += channel_overfit_eval
            epoch_term3 += model_complexity_eval
            epoch_channel_overfit_kl += channel_overfit_kl_val
            epoch_kl_ch_total += kl_ch_total_val
            epoch_mixture_kl += mixture_kl_val
            epoch_component_expected_kl += component_expected_kl_val
            epoch_k_hat += k_hat_val
            epoch_channel_penalty += channel_penalty_val

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
            
    return model, history

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
# MAIN EXECUTION: TEST14 CHANNEL SWEEP
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING TEST14: TRAIN CHANNEL OUTAGE PROBABILITY (P_tr) SWEEP")
    print(f"Fixed Test Outage Probability (P_te) = {P_OUTAGE_TE}")
    print("Evaluating Proposed Bound Regularization under mismatch scenario.")
    print("="*70 + "\n")

    # Set master seed
    set_seed(SEED)

    # Load moons dataset safely
    train_loader, test_loader, n_trains = get_dataloaders(
        N_SAMPLES,
        batch_size=BATCH_SIZE,
        seed=SEED,
        noise=MOONS_NOISE,
    )

    # Outage Probability sweep configuration
    p_tr_sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    EVAL_REPEATS = 10
    INFERENCE_WEIGHT_SAMPLES = 50

    # Tracking metrics across the sweep
    sweep_results = []
    
    for idx, p_tr in enumerate(p_tr_sweep):
        print(f"\n[{idx + 1}/{len(p_tr_sweep)}] Sweeping P_tr = {p_tr:.2f} ...")
        
        # 1. Update Global Outage Parameters dynamically
        P_OUTAGE_TR = p_tr
        
        # Re-evaluate dependent formulas based on new P_OUTAGE_TR
        KL_CH_TOTAL = channel_kl_total(HIDDEN_DIM)
        CHANNEL_PENALTY_CACHE.clear()  # Clear cache to guarantee correct calculations
        CH_PENALTY = get_channel_penalty(HIDDEN_DIM)
        
        print(f"   Dynamic D(P_ch || P_art) recalculated: {KL_CH_TOTAL:.4f}")
        print(f"   Dynamic E[||M - I||] recalculated: {CH_PENALTY:.4f}")

        # 2. Setup run-specific output directory for each sweep configuration
        run_dir = os.path.join(RESULTS_DIR, f"p_tr_{p_tr:.2f}")
        os.makedirs(run_dir, exist_ok=True)

        # 3. Train the Proposed Regularization model
        model_prop, history_prop = train_scenario(
            scenario_name='proposed',
            loader=train_loader,
            n_samples=n_trains,
            mode='train' if p_tr > 0 else 'perfect',
            objective='heuristic',
            seed=SEED,
            use_cache=True,
            run_dir=run_dir,
            verbose=True  # Keep console logs brief
        )

        # 4. Evaluate on Inference/Test channel under the mismatch
        test_loss, test_acc = evaluate_inference(
            model=model_prop,
            loader=test_loader,
            seed=SEED,
            repeats=EVAL_REPEATS,
            weight_mc_samples=INFERENCE_WEIGHT_SAMPLES,
        )

        train_loss = history_prop['train_loss'][-1]
        train_acc = history_prop['train_acc'][-1]

        # Log to terminal immediately
        print(f"   Train Performance -> Loss: {train_loss:.4f} | Accuracy: {train_acc*100:.2f}%")
        print(f"   Test Performance  -> Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%")

        # Save record
        sweep_results.append({
            "p_tr": p_tr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        # Garbage collection
        free_memory(model_prop)

    # Save the sweep summary configuration in a JSON file
    summary_path = os.path.join(RESULTS_DIR, "sweep_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "p_te_fixed": P_OUTAGE_TE,
            "sweep_data": sweep_results
        }, f, indent=4)
    print(f"\nSummary dictionary successfully saved to {summary_path}")

    # Plotting Accuracy vs P_tr
    p_tr_values = [res["p_tr"] for res in sweep_results]
    train_acc_values = [res["train_acc"] for res in sweep_results]
    test_acc_values = [res["test_acc"] for res in sweep_results]

    plt.figure(figsize=(10, 6))
    plt.plot(p_tr_values, test_acc_values, marker='o', color='teal', label='Test Inference Acc (at P_te=0.5)', linewidth=2.5)
    plt.plot(p_tr_values, train_acc_values, marker='x', linestyle='--', color='salmon', label='Training Accuracy', linewidth=1.5)
    plt.axvline(x=P_OUTAGE_TE, color='navy', linestyle=':', label=f'Matched Channel (P_tr = P_te = {P_OUTAGE_TE})', linewidth=2)
    
    plt.title('Inference Robustness Sweep (test14)', fontsize=14, fontweight='bold')
    plt.xlabel('Train Channel Outage Probability ($P_{tr}$)', fontsize=12)
    plt.ylabel('Model Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=11)
    plt.ylim(0.5, 1.05)
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "mismatch_robustness_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Robustness illustration saved to {plot_path}")

    # Generate Terminal visual alignment table
    print("\n" + "="*80)
    print("FINAL INFERENCE RESULTS METRIC MATRIX (test14)")
    print("="*80)
    print(f"{'Train Outage (P_tr)':<20} | {'Train Loss':>12} | {'Train Acc':>12} | {'Test Loss (P_te=0.5)':>20} | {'Test Acc (P_te=0.5)':>20}")
    print("-" * 100)
    for res in sweep_results:
        print(f"{res['p_tr']:<20.2f} | {res['train_loss']:>12.4f} | {res['train_acc']:>11.2%} | {res['test_loss']:>20.4f} | {res['test_acc']:>19.2%}")
    print("="*80)