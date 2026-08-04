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
RESULTS_DIR = os.path.join('results', 'test16')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 1. HARDWARE & PATH CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 2. HYPERPARAMETERS & CHANNEL DISTRIBUTIONS
# ==========================================
N_SAMPLES = 1000       # Total samples in the dataset
N_U_SETS = 10          # Number of independently sampled U sets / posterior components
HIDDEN_DIM = 64        # Network hidden dimension
IN_DIM = 2
OUT_DIM = 2
MOONS_NOISE = 0.3      # make_moons noise level
BATCH_SIZE = 64        # Batch size for training
EPOCHS = 150           # Training epochs
LR_BASE = 0.01        # Base learning rate
LR_DECAY_STEP = 75     # StepLR decay period (epochs)
LR_DECAY_GAMMA = 0.8   # StepLR decay factor
PRIOR_LAMBDA = 1.0     # Variance of isotropic Gaussian prior
EPSILON = 0.025        # PAC-Bayes confidence parameter
SIGMA_SQ = 1.0         # Assumed sub-Gaussian parameter
SIGMA_ART_SQ = 1.0     # Assumed sub-Gaussian parameter for artificial-channel loss
ALPHA_COEFF = 0.0      # Weighting factor for channel-shifting term
BETA_COEFF = 0.1       # Weighting factor for channel-overfitting term
GAMMA_COEFF = 0.001     # Weighting factor for PAC-Bayes bound term
M_ARTIFICIAL_CHANNELS_DEFAULT = 100  # Default m value
M_ARTIFICIAL_CHANNELS_LIST = [5, 10, 20, 50, 100, 200, 500, 1000] # m sizes to sweep
MI_MC_SAMPLES = 100     # MC samples for mixture KL
SEED = 42
LIPSCHITZ_METHOD_PERFECT = "grad"

P_OUTAGE_TE = 0.5
P_OUTAGE_TR = P_OUTAGE_TE  

def kl_bernoulli(p_te, p_tr, eps=1e-6):
    """Closed form KL D(Bern(p_te) || Bern(p_tr)) where p is probability of outage."""
    p_te_c = np.clip(p_te, eps, 1.0 - eps)
    p_tr_c = np.clip(p_tr, eps, 1.0 - eps)
    kl = p_te_c * np.log(p_te_c / p_tr_c) + (1 - p_te_c) * np.log((1 - p_te_c) / (1 - p_tr_c))
    return kl

def channel_kl_total(hidden_dim):
    """Total channel shift KL across independent hidden dimensions."""
    return hidden_dim * kl_bernoulli(P_OUTAGE_TE, P_OUTAGE_TR)

KL_CH_TOTAL = channel_kl_total(HIDDEN_DIM)

def estimate_expected_channel_norm(hidden_dim, p_outage_te, norm_type="frobenius", device="cpu"):
    """Calculates theoretical expected norm E[||M - I||] of a Bernoulli diagonal channel mask M."""
    if not (0.0 <= p_outage_te <= 1.0):
        raise ValueError(f"p_outage_te must be in [0, 1], got {p_outage_te}")

    n = int(hidden_dim)
    p_out = float(p_outage_te)
    norm_clean = norm_type.strip().lower()

    if p_out == 0.0:
        return torch.tensor(0.0, device=device)

    if norm_clean in ["spectral", "l2", "2"]:
        prob_entry_is_zero = torch.tensor(1.0 - p_out, device=device)
        prob_all_zero = torch.pow(prob_entry_is_zero, n)
        return 1.0 - prob_all_zero

    elif norm_clean in ["frobenius", "frob"]:
        k_values = torch.arange(n + 1, device=device)
        binom = dist.Binomial(total_count=n, probs=torch.tensor(p_out, device=device))
        log_probs = binom.log_prob(k_values)
        pmf = torch.exp(log_probs)
        return torch.sum(torch.sqrt(k_values) * pmf)

    else:
        raise ValueError(f"Unknown norm_type '{norm_type}'. Choose 'frobenius' or 'spectral'.")

CHANNEL_PENALTY_CACHE = {}

def get_channel_penalty(hidden_dim):
    if hidden_dim not in CHANNEL_PENALTY_CACHE:
        CHANNEL_PENALTY_CACHE[hidden_dim] = estimate_expected_channel_norm(
            hidden_dim, P_OUTAGE_TE, norm_type='frobenius', device=device
        )
    return CHANNEL_PENALTY_CACHE[hidden_dim]

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
        if mp.current_process().name == 'MainProcess':
            print(f"Loading existing data from {dataset_file}...")
        X_tensor, y_tensor = torch.load(dataset_file)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_size

def visualize_and_save_dataset(X, y, filename="dataset_viz.png", title="Dataset Visualization"):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
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
                self.num_artificial_channels, (self.K, B), generator=self.train_gen, device='cpu'
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

def evaluate_model(model, loader, mode='train'):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        theta_mean = model.mu if getattr(model, "is_bnn", False) else None
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = model(batch_x, theta_mean, mode=mode)
            y_expanded = batch_y.unsqueeze(0).expand(model.K, -1)
            loss = F.cross_entropy(out.reshape(-1, model.out_dim), y_expanded.reshape(-1))
            preds = out.argmax(dim=-1)
            total_correct += (preds == y_expanded).sum().item() / model.K 

            total_loss += loss.item() * batch_y.size(0)
            total += batch_y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)

def scenario_label(scenario_name, mode, objective):
    return f"{scenario_name}_{mode}_{objective}"

def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

def build_param_config(scenario_name, mode, objective, n_samples, seed, mi_mc_samples, lipschitz_method_perfect, batch_size=None, moon_noise=None, epochs=None, lr=None, lr_decay_step=None, lr_decay_gamma=None, alpha_coeff=None, beta_coeff=None, gamma_coeff=None, hidden_dim=None, n_u_sets=None, m_artificial_channels=None, is_bnn=True):
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
    m_artificial_channels = M_ARTIFICIAL_CHANNELS_DEFAULT if m_artificial_channels is None else m_artificial_channels
    return {
        "scenario_name": scenario_name, "mode": mode, "objective": objective,
        "n_samples": n_samples, "seed": seed,
        "batch_size": batch_size, "moon_noise": moon_noise,
        "is_bnn": is_bnn,
        "training": {
            "epochs": epochs,
            "lr": lr, "lr_decay_step": lr_decay_step, "lr_decay_gamma": lr_decay_gamma,
            "alpha_coeff": alpha_coeff, "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
            "hidden_dim": hidden_dim, "n_u_sets": n_u_sets,
            "m_artificial_channels": m_artificial_channels,
        }
    }

def get_run_dir(scenario_name, mode, objective, n_samples, seed, mi_mc_samples, lipschitz_method_perfect, batch_size=None, moon_noise=None, epochs=None, lr=None, lr_decay_step=None, lr_decay_gamma=None, alpha_coeff=None, beta_coeff=None, gamma_coeff=None, hidden_dim=None, n_u_sets=None, m_artificial_channels=None, is_bnn=True):
    param_cfg = build_param_config(scenario_name, mode, objective, n_samples, seed, mi_mc_samples, lipschitz_method_perfect, batch_size=batch_size, moon_noise=moon_noise, epochs=epochs, lr=lr, lr_decay_step=lr_decay_step, lr_decay_gamma=lr_decay_gamma, alpha_coeff=alpha_coeff, beta_coeff=beta_coeff, gamma_coeff=gamma_coeff, hidden_dim=hidden_dim, n_u_sets=n_u_sets, m_artificial_channels=m_artificial_channels, is_bnn=is_bnn)
    run_id = config_hash(param_cfg)
    label = scenario_label(scenario_name, mode, objective)
    run_dir = os.path.join(RESULTS_DIR, label, f"m_{m_artificial_channels}_param_{run_id}")
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
    if final_acc < 0.55 or final_loss > 5.0:
        return False
    return True

def save_training_history(run_dir, scenario_name, mode, objective, batch_size, moon_noise, n_samples, epochs, history, lr, lr_decay_step, lr_decay_gamma, alpha_coeff, beta_coeff, gamma_coeff, m_artificial_channels, inference=None):
    payload = {
        "scenario": scenario_name, "mode": mode, "objective": objective,
        "batch_size": batch_size, "moon_noise": moon_noise, "n_samples": n_samples, "epochs": epochs,
        "lr": lr, "lr_decay_step": lr_decay_step, "lr_decay_gamma": lr_decay_gamma,
        "alpha": alpha_coeff, "beta": beta_coeff, "gamma": gamma_coeff,
        "m_artificial_channels": m_artificial_channels,
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

def train_scenario(scenario_name, loader, n_samples, mode='train', objective='heuristic', batch_size=None, moon_noise=None, p_outage_te=None, p_outage_tr=None, epochs=None, lr=None, lr_decay_step=None, lr_decay_gamma=None, alpha_coeff=None, beta_coeff=None, gamma_coeff=None, mi_mc_samples=None, seed=None, lipschitz_method_perfect=None, hidden_dim=None, n_u_sets=None, m_artificial_channels=None, use_cache=True, run_dir=None, verbose=False):
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
    m_artificial_channels = M_ARTIFICIAL_CHANNELS_DEFAULT if m_artificial_channels is None else m_artificial_channels
    kl_ch_total = channel_kl_total(hidden_dim)
    channel_penalty = get_channel_penalty(hidden_dim)
    seed = SEED if seed is None else seed
    set_seed(seed)

    model = VectorizedBNNEnsemble(n_u_sets, IN_DIM, hidden_dim, OUT_DIM, m_artificial_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_decay_step and lr_decay_step > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    mi_mc_samples = MI_MC_SAMPLES if mi_mc_samples is None else mi_mc_samples
    lipschitz_method_perfect = LIPSCHITZ_METHOD_PERFECT if lipschitz_method_perfect is None else lipschitz_method_perfect

    if run_dir is None and use_cache:
        run_dir = get_run_dir(
            scenario_name, mode, objective, n_samples, seed, mi_mc_samples, lipschitz_method_perfect,
            batch_size=batch_size, moon_noise=moon_noise, epochs=epochs, lr=lr, lr_decay_step=lr_decay_step,
            lr_decay_gamma=lr_decay_gamma, alpha_coeff=alpha_coeff, beta_coeff=beta_coeff, gamma_coeff=gamma_coeff,
            hidden_dim=hidden_dim, n_u_sets=n_u_sets, m_artificial_channels=m_artificial_channels, is_bnn=True,
        )
    else:
        run_dir = run_dir or "temp_run"

    weights_path = os.path.join(run_dir, "weights.pth")

    if use_cache and os.path.exists(weights_path):
        print(f"Loading existing weights for m={m_artificial_channels}: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        with open(os.path.join(run_dir, "history.json"), "r", encoding="utf-8") as f:
            history = json.load(f).get("history", {})
        return model, history

    complexity_term = np.log(np.sqrt(n_samples) / EPSILON)

    history = {
        'train_loss': [], 'train_acc': [], 'bound_total': [],
        'bound_term1': [], 'bound_term2': [], 'bound_term3': [],
        'joint_complexity': [], 'channel_overfit_kl': [], 'kl_ch_total': [],
        'mixture_kl': [], 'component_expected_kl': [], 'k_hat': [],
        'channel_penalty': [], 'k_hat_channel_penalty': [], 'applied_regularization': [],
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
        epoch_joint_complexity = 0.0
        epoch_kl_ch_total = 0.0
        epoch_component_expected_kl = 0.0

        model.train()
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            theta = model.sample_theta(1).squeeze(0)
            out = model(batch_x, theta, mode=mode)
            y_expanded = batch_y.unsqueeze(0).expand(model.K, -1)
            ce_loss = F.cross_entropy(out.reshape(-1, OUT_DIM), y_expanded.reshape(-1))

            component_expected_kl = compute_component_expected_kl(model)
            component_expected_kl = torch.clamp(component_expected_kl, min=0)
            component_expected_kl_val = component_expected_kl.item()

            channel_shift = ce_loss.new_tensor(np.sqrt(2 * SIGMA_ART_SQ * kl_ch_total))
            channel_overfit = torch.sqrt((2 * SIGMA_ART_SQ / (m_artificial_channels - 1)) * (component_expected_kl + np.log(np.sqrt(m_artificial_channels) / EPSILON)))
            model_complexity = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (component_expected_kl + complexity_term))
            joint_complexity = torch.sqrt((2 * SIGMA_ART_SQ / (m_artificial_channels - 1) + 2 * SIGMA_SQ / (n_samples - 1)) * (component_expected_kl + np.log(np.sqrt(m_artificial_channels*n_samples) / EPSILON)))

            channel_shift_eval = channel_shift.item()
            channel_overfit_eval = channel_overfit.item()
            model_complexity_eval = model_complexity.item()
            joint_complexity_eval = joint_complexity.item()
            kl_ch_total_val = kl_ch_total

            bound = channel_shift + joint_complexity
            bound_val = bound.item()

            if objective == 'bound':
                loss = ce_loss + bound
            elif objective == 'heuristic':
                loss = ce_loss + (alpha_coeff * channel_shift + gamma_coeff * joint_complexity)
            else:
                loss = ce_loss

            if not torch.isfinite(loss):
                print(f"Stopping early: non-finite loss at epoch {epoch + 1}.")
                history['nonfinite_loss'] = True
                stop_training = True
                break

            loss.backward()
            optimizer.step()
            
            epoch_loss += ce_loss.item()
            epoch_reg += loss.item() - ce_loss.item()
            epoch_bound_total += bound_val
            epoch_term1 += channel_shift_eval
            epoch_term2 += channel_overfit_eval
            epoch_term3 += model_complexity_eval
            epoch_joint_complexity += joint_complexity_eval
            epoch_kl_ch_total += kl_ch_total_val
            epoch_component_expected_kl += component_expected_kl_val

        train_loss, train_acc = evaluate_model(model, loader, mode=mode)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['bound_total'].append(epoch_bound_total / len(loader))
        history['bound_term1'].append(epoch_term1 / len(loader))
        history['bound_term2'].append(epoch_term2 / len(loader))
        history['bound_term3'].append(epoch_term3 / len(loader))
        history['joint_complexity'].append(epoch_joint_complexity / len(loader))
        history['kl_ch_total'].append(epoch_kl_ch_total / len(loader))
        history['component_expected_kl'].append(epoch_component_expected_kl / len(loader))
        history['applied_regularization'].append(epoch_reg / len(loader))

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"m={m_artificial_channels} | Epoch {epoch+1}/{epochs} | "
                f"CE Loss: {epoch_loss/len(loader):.4f} | "
                f"Joint Comp: {epoch_joint_complexity/len(loader):.4f} | "
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

    if use_cache:
        torch.save(model.state_dict(), weights_path)
        save_training_history(
            run_dir, scenario_name, mode, objective, batch_size, moon_noise, n_samples,
            epochs, history, lr, lr_decay_step, lr_decay_gamma, alpha_coeff, beta_coeff, gamma_coeff,
            m_artificial_channels
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
    beta_coeff = task_config['beta_coeff']
    gamma_coeff = task_config['gamma_coeff']
    n_u_sets = task_config['n_u_sets']

    set_seed(seed)
    
    train_loader, test_loader, n_trains = get_dataloaders(
        n_samples, batch_size=batch_size, seed=seed, noise=moon_noise,
    )

    model, history = train_scenario(
        'proposed', train_loader, n_trains, mode='train', objective='heuristic',
        hidden_dim=hidden_dim, batch_size=batch_size, moon_noise=moon_noise, p_outage_te=p_outage_te, p_outage_tr=p_outage_tr, epochs=epochs,
        n_u_sets=n_u_sets, m_artificial_channels=m_artificial_channels,
        lr=lr, lr_decay_step=lr_decay_step, lr_decay_gamma=lr_decay_gamma, alpha_coeff=0.0, beta_coeff=beta_coeff, gamma_coeff=gamma_coeff,
        seed=seed, use_cache=use_cache, verbose=verbose
    )
    loss, acc = evaluate_inference(model, test_loader, seed=seed, repeats=repeats, weight_mc_samples=weight_mc_samples)
    
    final_train_loss = history["train_loss"][-1] if history.get("train_loss") else None
    final_train_acc = history["train_acc"][-1] if history.get("train_acc") else None
    final_bound_total = history["bound_total"][-1] if history.get("bound_total") else None
    final_joint_complexity = history["joint_complexity"][-1] if history.get("joint_complexity") else None
    
    free_memory(model)
    
    return {
        "seed": seed, "scenario": "proposed", "objective": "heuristic", "mode": "train",
        "hidden_dim": hidden_dim, "batch_size": batch_size, "moon_noise": moon_noise,
        "p_outage_te": p_outage_te, "p_outage_tr": p_outage_tr, "n_samples": n_samples, "epochs": epochs,
        "n_u_sets": n_u_sets, "m_artificial_channels": m_artificial_channels,
        "lr": lr, "lr_decay_step": lr_decay_step, "lr_decay_gamma": lr_decay_gamma,
        "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
        "loss": loss, "acc": acc, "train_loss": final_train_loss, "train_acc": final_train_acc,
        "bound_total": final_bound_total, "joint_complexity": final_joint_complexity
    }


# ==========================================
# PLOTTING UTILITIES FOR M SWEEP
# ==========================================
def plot_m_sweep_comparison(m_list, pop_accs, pop_losses, emp_losses, bound_totals, save_dir):
    """Generates visualization plots comparing performance across different m_artificial_channels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Population Accuracy vs m
    axes[0].plot(m_list, [acc * 100 for acc in pop_accs], 'o-', color='navy', linewidth=2, markersize=8, label='Population Acc (%)')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Artificial Channels m (log scale)', fontsize=12)
    axes[0].set_ylabel('Population Accuracy (%)', fontsize=12)
    axes[0].set_title('Effect of Artificial Channels (m) on Accuracy', fontsize=13)
    axes[0].grid(True, which="both", linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=11)

    # Plot 2: Losses & Bounds vs m
    axes[1].plot(m_list, pop_losses, 's-', color='crimson', linewidth=2, markersize=7, label='Pop. Loss (Test)')
    axes[1].plot(m_list, emp_losses, '^-', color='forestgreen', linewidth=2, markersize=7, label='Emp. Loss (Train)')
    axes[1].plot(m_list, bound_totals, 'd--', color='purple', linewidth=2, markersize=7, label='PAC-Bayes Bound')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Artificial Channels m (log scale)', fontsize=12)
    axes[1].set_ylabel('Loss / Bound Value', fontsize=12)
    axes[1].set_title('Effect of Artificial Channels (m) on Loss & Bound', fontsize=13)
    axes[1].grid(True, which="both", linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=11)

    fig.tight_layout()
    filename = os.path.join(save_dir, "m_artificial_channels_comparison.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison plot to {filename}")


def plot_m_bound_trajectories(m_histories, save_dir):
    """Plots the PAC-Bayes bound trajectories over training epochs for different m_artificial_channels on a single image."""
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(m_histories)))

    for idx, (m_val, history) in enumerate(m_histories.items()):
        bound_series = history.get('bound_total', [])
        if bound_series:
            epochs = list(range(1, len(bound_series) + 1))
            plt.plot(epochs, bound_series, label=f"m = {m_val}", color=colors[idx], linewidth=2.0)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PAC-Bayes Bound', fontsize=12)
    plt.title('PAC-Bayes Bound Dynamics Across Training Epochs for Different m', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Artificial Channels (m)", fontsize=10, title_fontsize=11, loc='upper right')
    plt.tight_layout()

    filename = os.path.join(save_dir, "m_bound_trajectories.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bound trajectory plot across m to {filename}")


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
            N_SAMPLES, batch_size=BATCH_SIZE, seed=SEED, noise=MOONS_NOISE,
        )

        results = []
        m_histories = {}
        print("\n" + "="*75)
        print(f"RUNNING M_ARTIFICIAL_CHANNELS SWEEP FOR PROPOSED METHOD (Seed: {SEED})")
        print("="*75)
        print(f"{'m_channels':<12} {'Pop. Loss':>10} {'Pop. Acc (%)':>14} {'Emp. Loss':>12} {'Bound Total':>14}")
        print("-" * 75)

        pop_losses = []
        pop_accs = []
        emp_losses = []
        bound_totals = []

        for m_channels in M_ARTIFICIAL_CHANNELS_LIST:
            model_prop, history_prop = train_scenario(
                'proposed', train_loader, n_trains, mode='train', objective='heuristic',
                seed=SEED, m_artificial_channels=m_channels, use_cache=True, verbose=False
            )
            loss_prop, acc_prop = evaluate_inference(
                model_prop, test_loader, seed=SEED, repeats=EVAL_REPEATS, weight_mc_samples=INFERENCE_WEIGHT_SAMPLES
            )
            
            emp_loss = history_prop['train_loss'][-1]
            bound_tot = history_prop['bound_total'][-1]

            pop_losses.append(loss_prop)
            pop_accs.append(acc_prop)
            emp_losses.append(emp_loss)
            bound_totals.append(bound_tot)
            m_histories[m_channels] = history_prop

            print(f"{m_channels:<12} {loss_prop:>10.4f} {acc_prop*100:>13.2f}% {emp_loss:>12.4f} {bound_tot:>14.4f}")

            results.append({
                "m_artificial_channels": m_channels,
                "pop_loss": loss_prop,
                "pop_acc": acc_prop,
                "emp_loss": emp_loss,
                "bound_total": bound_tot,
            })
            free_memory(model_prop)

        print("="*75)

        # Plot single-run comparison graph
        plot_m_sweep_comparison(M_ARTIFICIAL_CHANNELS_LIST, pop_accs, pop_losses, emp_losses, bound_totals, RESULTS_DIR)

        # Plot bound trajectory curves across epochs for different m on a single image
        plot_m_bound_trajectories(m_histories, RESULTS_DIR)

        # Save single-run summary JSON
        summary_path = os.path.join(RESULTS_DIR, "m_sweep_single_run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Saved single run summary to {summary_path}")

    else:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        results_jsonl_path = os.path.join(RESULTS_DIR, "sweep_results.jsonl")
        with open(results_jsonl_path, "w", encoding="utf-8") as f:
            pass 

        MASTER_SEED = 42
        random.seed(MASTER_SEED)

        num_seeds = 10
        SEEDS = [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]
        HIDDEN_DIM_GRID = [64]
        N_U_SETS_GRID = [10]
        M_ARTIFICIAL_CHANNELS_GRID = [5, 10, 20, 50, 100, 200, 500, 1000]
        LR_GRID = [0.01, 0.005, 0.001]
        LR_DECAY_STEP_GRID = [75]
        LR_DECAY_GAMMA_GRID = [0.8]
        BETA_COEFF_GRID = [0.1]
        GAMMA_COEFF_GRID = [0.05, 0.01]
        EPOCHS_GRID = [150]
        use_cache = False
        verbose = False

        print("Pre-generating dataset .pt files...")
        for seed in SEEDS:
            get_dataloaders(N_SAMPLES, batch_size=BATCH_SIZE, seed=seed, noise=MOONS_NOISE)

        print("Building task queue for Proposed Method m-sweep...")
        tasks = []
        for seed in SEEDS:
            for hidden_dim in HIDDEN_DIM_GRID:
                for n_u_sets in N_U_SETS_GRID:
                    for m_artificial_channels in M_ARTIFICIAL_CHANNELS_GRID:
                        for epochs in EPOCHS_GRID:
                            for lr in LR_GRID:
                                for lr_decay_step in LR_DECAY_STEP_GRID:
                                    for lr_decay_gamma in LR_DECAY_GAMMA_GRID:
                                        for beta_coeff in BETA_COEFF_GRID:
                                            for gamma_coeff in GAMMA_COEFF_GRID:
                                                tasks.append({
                                                    "scenario": "proposed", "seed": seed, "hidden_dim": hidden_dim,
                                                    "batch_size": BATCH_SIZE, "moon_noise": MOONS_NOISE, "n_samples": N_SAMPLES,
                                                    "epochs": epochs, "n_u_sets": n_u_sets,
                                                    "m_artificial_channels": m_artificial_channels, 
                                                    "lr": lr, "lr_decay_step": lr_decay_step, "lr_decay_gamma": lr_decay_gamma,
                                                    "beta_coeff": beta_coeff, "gamma_coeff": gamma_coeff,
                                                    "use_cache": use_cache, "verbose": verbose,
                                                })

        print(f"Total proposed method sweep tasks generated: {len(tasks)}")

        MAX_WORKERS = min(16, max(1, os.cpu_count() - 1))
        print(f"Starting parallel execution with {MAX_WORKERS} workers...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(run_sweep_task, task, EVAL_REPEATS, INFERENCE_WEIGHT_SAMPLES): task for task in tasks}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                completed += 1
                try:
                    result = future.result()
                    with open(results_jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result) + "\n")
                        
                    if completed % 20 == 0 or completed == len(tasks):
                        print(f"Progress: {completed}/{len(tasks)} tasks completed.")
                except Exception as exc:
                    print(f"A task generated an exception: {exc}")

        print(f"Sweep complete. Results saved to {results_jsonl_path}")

        # Post-processing sweep results grouped by m_artificial_channels
        sweep_results = []
        with open(results_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                sweep_results.append(json.loads(line.strip()))

        m_summary = {}
        for m_val in M_ARTIFICIAL_CHANNELS_GRID:
            m_res = [res for res in sweep_results if res['m_artificial_channels'] == m_val]
            if m_res:
                accs = [res['acc'] for res in m_res]
                losses = [res['loss'] for res in m_res]
                train_losses = [res['train_loss'] for res in m_res if res.get('train_loss') is not None]
                bounds = [res['bound_total'] for res in m_res if res.get('bound_total') is not None]
                best_task = max(m_res, key=lambda x: x['acc'])

                m_summary[str(m_val)] = {
                    "m_artificial_channels": m_val,
                    "num_experiments": len(m_res),
                    "mean_acc": float(np.mean(accs)),
                    "std_acc": float(np.std(accs)),
                    "max_acc": float(np.max(accs)),
                    "mean_loss": float(np.mean(losses)),
                    "std_loss": float(np.std(losses)),
                    "mean_train_loss": float(np.mean(train_losses)) if train_losses else None,
                    "mean_bound_total": float(np.mean(bounds)) if bounds else None,
                    "best_config": best_task
                }

        summary_path = os.path.join(RESULTS_DIR, "m_sweep_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(m_summary, f, indent=4)

        print(f"Summary by m saved to {summary_path}")

        # Plotting sweep aggregates
        m_vals_sorted = sorted([int(m) for m in m_summary.keys()])
        mean_accs = [m_summary[str(m)]["mean_acc"] for m in m_vals_sorted]
        mean_losses = [m_summary[str(m)]["mean_loss"] for m in m_vals_sorted]
        mean_train_losses = [m_summary[str(m)]["mean_train_loss"] for m in m_vals_sorted]
        mean_bounds = [m_summary[str(m)]["mean_bound_total"] for m in m_vals_sorted]

        plot_m_sweep_comparison(m_vals_sorted, mean_accs, mean_losses, mean_train_losses, mean_bounds, RESULTS_DIR)
