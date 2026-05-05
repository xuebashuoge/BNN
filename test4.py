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
RESULTS_DIR = os.path.join('results', 'test4')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
WEIGHTS_DIR = os.path.join(RESULTS_DIR, 'weights')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ==========================================
# 1. HARDWARE & PATH CONFIGURATION (CRITICAL)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 2. HYPERPARAMETERS & CHANNEL DISTRIBUTIONS
# ==========================================
N_SAMPLES = 1000       
K_ENSEMBLE = 100        
HIDDEN_DIM = 16       
IN_DIM = 2
OUT_DIM = 2
BATCH_SIZE = 64        
EPOCHS = 150           
PRIOR_LAMBDA = 1.0     
EPSILON = 0.05         
SIGMA_SQ = 1.0         
SIGMA_0_SQ = 1.0       
REG_COEFF = 0.05       
REG_ALPHA = 0.5        
REG_BETA = 0.5         

MU_M_TR, STD_M_TR = 1.0, 0.1  
MU_B_TR, STD_B_TR = 0.0, 0.1  
MU_M_TE, STD_M_TE = 0.5, 0.5   
MU_B_TE, STD_B_TE = 0.0, 1.5   

def kl_gaussian_1d(mu1, std1, mu2, std2):
    return np.log(std2/std1) + (std1**2 + (mu1 - mu2)**2) / (2 * std2**2) - 0.5

kl_ch_m = HIDDEN_DIM * kl_gaussian_1d(MU_M_TR, STD_M_TR, MU_M_TE, STD_M_TE)
kl_ch_b = HIDDEN_DIM * kl_gaussian_1d(MU_B_TR, STD_B_TR, MU_B_TE, STD_B_TE)
KL_CH_TOTAL = kl_ch_m + kl_ch_b
print(f"Constant D(P_art || P_ch) = {KL_CH_TOTAL:.4f}")

def estimate_expected_channel_norm(hidden_dim, mu_m_te, mu_b_te, std_m_te, std_b_te, norm_type='frobenius', mc_samples=1000, device='cpu'):
    M_diff = torch.randn(mc_samples, hidden_dim, hidden_dim, device=device) * std_m_te + mu_m_te - 1.0
    B = torch.randn(mc_samples, hidden_dim, 1, device=device) * std_b_te + mu_b_te
    W_diff = torch.cat([M_diff, B], dim=2)
    
    if norm_type == 'frobenius':
        norms = torch.linalg.matrix_norm(W_diff, ord='fro')
    elif norm_type == 'spectral':
        norms = torch.linalg.matrix_norm(W_diff, ord=2)
    else:
        raise ValueError("norm_type must be 'frobenius' or 'spectral'")
    return norms.mean()

CH_PENALTY = estimate_expected_channel_norm(HIDDEN_DIM, MU_M_TE, MU_B_TE, STD_M_TE, STD_B_TE, device=device)

# ==========================================
# 3. DATA GENERATION
# ==========================================
data_path = os.environ.get('DATASET', './data')
os.makedirs(data_path, exist_ok=True)

def get_dataloaders(n_samples, batch_size=32):
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
        
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    
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
    print(f"Visualization saved as {filename}")

# ==========================================
# 4. ENSEMBLE BNN & STOCHASTIC CHANNEL LAYER
# ==========================================
class StochasticChannelLayer(nn.Module):
    def __init__(self, K, hid_dim):
        super().__init__()
        self.K = K
        self.hid_dim = hid_dim
        self.train_gen = torch.Generator(device='cpu')
        self.train_gen.manual_seed(1337)

    def reset_seed(self):
        """Locks the sequence across epochs by resetting."""
        self.train_gen.manual_seed(1337)

    def forward(self, x, mode='perfect'):
        B = x.shape[1]
        
        if mode == 'perfect':
            m = torch.ones_like(x, device=x.device)
            b = torch.zeros_like(x, device=x.device)
        elif mode == 'train':
            m = torch.randn(self.K, B, self.hid_dim, generator=self.train_gen, device='cpu').to(x.device) * STD_M_TR + MU_M_TR
            b = torch.randn(self.K, B, self.hid_dim, generator=self.train_gen, device='cpu').to(x.device) * STD_B_TR + MU_B_TR
        elif mode == 'test':
            m = torch.randn(self.K, B, self.hid_dim, device=x.device) * STD_M_TE + MU_M_TE
            b = torch.randn(self.K, B, self.hid_dim, device=x.device) * STD_B_TE + MU_B_TE
        else:
            raise ValueError(f"Invalid mode '{mode}'.")
            
        return x * m + b

class VectorizedBNNEnsemble(nn.Module):
    def __init__(self, K, in_dim, hid_dim, out_dim):
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.D1 = in_dim * hid_dim + hid_dim
        self.D2 = hid_dim * out_dim + out_dim
        self.D = self.D1 + self.D2
        
        # FIX: Shared base initialization prevents MI logsumexp underflow 
        # By starting near each other, the gradients can pull/push actively.
        base_mu = torch.randn(1, self.D) * 0.1
        self.mu = nn.Parameter(base_mu.repeat(K, 1) + torch.randn(K, self.D) * 1e-4)
        
        base_rho = torch.randn(1, self.D) * 0.1 - 3.0
        self.rho = nn.Parameter(base_rho.repeat(K, 1))
        
        self.channel_layer = StochasticChannelLayer(K, hid_dim)

    def get_sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample_theta(self, num_samples=1):
        sigma = self.get_sigma()
        eps = torch.randn(num_samples, self.K, self.D, device=device)
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
# 5. MONTE CARLO MUTUAL INFORMATION CALCULATIONS 
# ==========================================
def compute_mc_mutual_information_and_kl(model, prior_lambda=PRIOR_LAMBDA, mc_samples=3):
    """
    Computes Mutual Information via Monte Carlo integration over the K ensemble members.
    Follows the theoretical framing that the K members represent the empirical mixture P_{W|S}.
    """
    mu = model.mu
    sigma = model.get_sigma()
    K, D = mu.shape
    
    # 1. Expected KL to Prior: E_k [ D(q_k || Q) ]
    var = sigma**2 + 1e-8
    kl_k = 0.5 * torch.sum(var/prior_lambda + mu**2/prior_lambda - 1 - torch.log(var/prior_lambda), dim=-1)
    expected_kl = kl_k.mean()
    
    if K == 1:
        return expected_kl, torch.tensor(0.0, device=device)
        
    # 2. Sample \tilde{W} from each conditional posterior
    # W_tilde shape: (mc_samples, K, D)
    eps = torch.randn(mc_samples, K, D, device=device)
    W_tilde = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    
    # 3. Compute Conditional Log Density: log P(\tilde{W} | W^{(l_0)})
    # We evaluate each sample under its OWN generating distribution (the specific ensemble member)
    log_scale_cond = torch.log(var) * 0.5 + 0.5 * np.log(2 * np.pi)
    diff_cond = (W_tilde - mu.unsqueeze(0))**2 / (2 * var.unsqueeze(0))
    log_prob_cond = -torch.sum(log_scale_cond.unsqueeze(0) + diff_cond, dim=-1) # Shape: (mc_samples, K)
    
    # 4. Compute Marginal Log Density: log P(\tilde{W} | S)
    # We evaluate each sample under the mixture of ALL K components
    # Expand dims to compute all pairwise densities: N(W_tilde_{s,k} | mu_j, sigma_j)
    W_tilde_exp = W_tilde.unsqueeze(2)        # (mc_samples, K_samples, 1, D)
    mu_exp = mu.unsqueeze(0).unsqueeze(0)     # (1, 1, K_components, D)
    var_exp = var.unsqueeze(0).unsqueeze(0)   # (1, 1, K_components, D)
    
    log_scale_marg = torch.log(var_exp) * 0.5 + 0.5 * np.log(2 * np.pi)
    diff_marg = (W_tilde_exp - mu_exp)**2 / (2 * var_exp)
    
    # log density of each sample under each component
    log_prob_all = -torch.sum(log_scale_marg + diff_marg, dim=-1) # Shape: (mc_samples, K_samples, K_components)
    
    # Use logsumexp over components (dim 2) for numerical stability, subtract log(K) for the mixture average
    log_prob_marg = torch.logsumexp(log_prob_all, dim=2) - np.log(K) # Shape: (mc_samples, K_samples)
    
    # 5. Expected Mutual Information: E[log P_cond - log P_marg]
    mi_samples = log_prob_cond - log_prob_marg
    mi = mi_samples.mean() 
    
    return expected_kl, mi

# ==========================================
# 6. TRAINING LOOP & EVALUATION
# ==========================================
def evaluate_model(model, loader, mode='test'):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        theta_mean = model.mu
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = model(batch_x, theta_mean, mode=mode)
            avg_logits = out.mean(dim=0)
            loss = F.cross_entropy(avg_logits, batch_y)
            probs = F.softmax(out, dim=-1)
            preds = probs.mean(dim=0).argmax(dim=-1)
            total_loss += loss.item() * batch_y.size(0)
            total_correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)

def plot_training_metrics(history, scenario_name, mode):
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
    fig.suptitle(f"{scenario_name.upper()} ({mode})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(PLOTS_DIR, f"metrics_{scenario_name}_{mode}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_bound_decomposition(history, scenario_name, mode):
    if not history.get('bound_total'): return
    epochs = list(range(1, len(history['bound_total']) + 1))
    series = [('Bound Total', history['bound_total']), ('Bound Term 1', history['bound_term1']), 
              ('Bound Term 2', history['bound_term2']), ('MI Bound', history.get('mi', []))]
    plotted = [(label, values) for label, values in series if values and max(values) != 0.0]
    if not plotted: return

    fig, axes = plt.subplots(len(plotted), 1, figsize=(10, 2.2 * len(plotted)), sharex=True)
    axes = [axes] if len(plotted) == 1 else axes
    for ax, (label, values) in zip(axes, plotted):
        ax.plot(epochs, values, label=label)
        ax.set_ylabel(label)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
    axes[-1].set_xlabel('Epoch')
    fig.suptitle(f"Bound Decomposition: {scenario_name.upper()} ({mode})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(PLOTS_DIR, f"bound_{scenario_name}_{mode}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def train_scenario(scenario_name, loader, n_samples, mode='perfect', objective='bound'):
    print(f"\n--- Training Scenario: {scenario_name.upper()} ---")
    # model = VectorizedBNNEnsemble(1 if scenario_name == 'erm' or mode == 'perfect' else K_ENSEMBLE, IN_DIM, HIDDEN_DIM, OUT_DIM).to(device)
    model = VectorizedBNNEnsemble(K_ENSEMBLE, IN_DIM, HIDDEN_DIM, OUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    complexity_term = np.log(np.sqrt(n_samples) / EPSILON)
    
    history = {k: [] for k in ['train_loss', 'train_acc', 'bound_total', 'bound_term1', 'bound_term2', 'mi']}

    for epoch in range(EPOCHS):
        model.channel_layer.reset_seed() # Locks channel sequences per epoch
        
        epoch_loss, epoch_reg, epoch_t1, epoch_t2, epoch_mi = 0.0, 0.0, 0.0, 0.0, 0.0
        model.train()
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            theta = model.sample_theta(1).squeeze(0)
            out = model(batch_x, theta, mode=mode)
            y_expanded = batch_y.unsqueeze(0).expand(model.K, -1)
            ce_loss = F.cross_entropy(out.reshape(-1, OUT_DIM), y_expanded.reshape(-1))
            
            loss, reg_val, term1_val, term2_val, mi_val = ce_loss, 0.0, 0.0, 0.0, 0.0
            
            if scenario_name in ['pac_bayes', 'proposed']:
                # Replaced analytical bound with rigorous Monte Carlo estimate
                expected_kl, mi_mc = compute_mc_mutual_information_and_kl(model, mc_samples=3)
                
                if scenario_name == 'pac_bayes':
                    reg = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (torch.clamp(expected_kl, min=0) + complexity_term))
                    loss = ce_loss + (REG_COEFF * reg)
                    reg_val = term1_val = reg.item()
                    
                elif scenario_name == 'proposed':
                    if mode == 'perfect':
                        grad_theta = torch.autograd.grad(ce_loss, theta, create_graph=True, retain_graph=True)[0]
                        K_hat = torch.norm(grad_theta, p=2)
                        term1 = K_hat * CH_PENALTY
                        term2 = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (expected_kl + complexity_term))
                        reg = term1 + term2
                        loss = ce_loss + (REG_COEFF * reg if objective == 'bound' else REG_COEFF * (REG_ALPHA * term1 + REG_BETA * expected_kl))
                        term1_val, term2_val, reg_val = term1.item(), term2.item(), reg.item()
                    elif mode == 'train':
                        term1 = torch.sqrt(2 * SIGMA_0_SQ * (torch.clamp(mi_mc, min=0) + KL_CH_TOTAL))
                        term2 = torch.sqrt((2 * SIGMA_SQ / (n_samples - 1)) * (expected_kl + complexity_term))
                        reg = term1 + term2
                        loss = ce_loss + (REG_COEFF * reg if objective == 'bound' else REG_COEFF * (REG_ALPHA * mi_mc + REG_BETA * expected_kl))
                        term1_val, term2_val, mi_val, reg_val = term1.item(), term2.item(), mi_mc.item(), reg.item()

            loss.backward()
            optimizer.step()
            
            epoch_loss += ce_loss.item()
            epoch_reg += reg_val 
            epoch_t1 += term1_val
            epoch_t2 += term2_val
            epoch_mi += mi_val

        train_loss, train_acc = evaluate_model(model, loader, mode=mode)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['bound_total'].append(epoch_reg / len(loader))
        history['bound_term1'].append(epoch_t1 / len(loader))
        history['bound_term2'].append(epoch_t2 / len(loader))
        history['mi'].append(epoch_mi / len(loader))

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | CE: {epoch_loss/len(loader):.4f} | Reg: {epoch_reg/len(loader):.4f} | MI: {epoch_mi/len(loader):.4f} | Acc: {train_acc*100:.2f}%")

    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"weights_{scenario_name}_{mode}.pth"))
    plot_training_metrics(history, scenario_name, mode)
    plot_bound_decomposition(history, scenario_name, mode)
    return model

def evaluate_inference(model, loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        theta_mean = model.mu
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = model(batch_x, theta_mean, mode='test')
            loss = F.cross_entropy(out.mean(dim=0), batch_y)
            preds = F.softmax(out, dim=-1).mean(dim=0).argmax(dim=-1)
            total_loss += loss.item() * batch_y.size(0)
            total_correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)

if __name__ == "__main__":
    train_loader, test_loader, n_trains = get_dataloaders(N_SAMPLES, batch_size=BATCH_SIZE)
    
    model_erm_perfect = train_scenario('erm', train_loader, n_trains, mode='perfect')
    loss_erm_perfect, acc_erm_perfect = evaluate_inference(model_erm_perfect, test_loader)

    model_erm = train_scenario('erm', train_loader, n_trains, mode='train')
    loss_erm, acc_erm = evaluate_inference(model_erm, test_loader)
    
    model_prop = train_scenario('proposed', train_loader, n_trains, mode='train')
    loss_prop, acc_prop = evaluate_inference(model_prop, test_loader)

    model_prop_perfect = train_scenario('proposed', train_loader, n_trains, mode='perfect')
    loss_prop_perfect, acc_prop_perfect = evaluate_inference(model_prop_perfect, test_loader)
    
    print("\n" + "="*50)
    print("FINAL INFERENCE RESULTS (EVALUATED ON P_ch)")
    print("="*50)
    print(f"Standard ERM + Perfect Channel Loss/Acc: {loss_erm_perfect:.4f} / {acc_erm_perfect*100:.2f}%")
    print(f"Standard ERM Loss/Acc:                {loss_erm:.4f} / {acc_erm*100:.2f}%")
    print(f"Proposed Bound Reg Loss/Acc:          {loss_prop:.4f} / {acc_prop*100:.2f}%")
    print(f"Proposed Bound + Perfect Channel Loss/Acc: {loss_prop_perfect:.4f} / {acc_prop_perfect*100:.2f}%")
    print("="*50)