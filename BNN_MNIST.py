import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# 1. Straight-Through Estimators (STE) for Binarization
# ---------------------------------------------------------
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # Sign function: returns -1 or 1. (Map 0 to 1 to avoid zeros)
        sign = torch.sign(x)
        sign = torch.where(sign == 0, torch.tensor(1.0, device=x.device), sign)
        return sign

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # HardTanh clipping for STE: cancel gradients if values get too large
        grad_input[x.abs() > 1.0] = 0
        return grad_input

# ---------------------------------------------------------
# 2. Custom Neural Network Layers
# ---------------------------------------------------------
class BinarizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, input):
        # Apply STE Binarization to weights
        bin_weight = BinarizeSTE.apply(self.weight)
        
        # Apply STE Binarization to bias if it exists
        bin_bias = None
        if self.bias is not None:
            bin_bias = BinarizeSTE.apply(self.bias)
            
        return F.linear(input, bin_weight, bin_bias)

class DiscreteChannelLayer(nn.Module):
    def __init__(self, use_stochastic_channel=True, threshold=28, p_high=0.99, p_low=0.01, temp=5.0):
        super().__init__()
        self.p_high = p_high
        self.p_low = p_low
        self.use_stochastic_channel = use_stochastic_channel
        self.threshold = threshold
        self.temp = temp
        # State to track the percentage of positive activations
        self.last_target_ratio = 0.0

    def forward(self, x):
        # Tracking positivity ratio of the input for visualization
        with torch.no_grad():
            target_mask = (x >= self.threshold).float()
            self.last_target_ratio = target_mask.mean().item()

        if not self.use_stochastic_channel:
            return x
        
        # # Probability of M_i = 1 depends on the whether it is greater than the standard deviation threshold (28) for input, which ranges from 0 to 784
        # p_matrix = torch.where(x >= self.threshold, torch.tensor(self.p_high, device=x.device), torch.tensor(self.p_low, device=x.device))

        # # Sample M from the probability distribution
        # M = torch.bernoulli(p_matrix)

        # --- STE for the Step Function ---
        # Smooth sigmoid provides a healthy gradient pushing x towards/past the threshold
        p_soft = self.p_low + (self.p_high - self.p_low) * torch.sigmoid((x - self.threshold) / self.temp)
        # Hard logic used for the actual forward pass
        p_hard = torch.where(x >= self.threshold, torch.tensor(self.p_high, device=x.device), torch.tensor(self.p_low, device=x.device))
        
        # Reparameterization Trick: p_hard on forward, p_soft on backward
        p_matrix = p_soft + (p_hard - p_soft).detach()

        # --- STE for the Bernoulli Sampling ---
        # Hard discrete sampling for forward pass
        M_hard = torch.bernoulli(p_matrix)
        # Reparameterization Trick: M_hard on forward, pass through p_matrix gradient on backward
        M = p_matrix + (M_hard - p_matrix).detach()

        return M * x

        

# ---------------------------------------------------------
# 3. Network Architecture
# ---------------------------------------------------------
class MNISTChannelNet(nn.Module):
    def __init__(self, d=512, use_stochastic_channel=True):
        super().__init__()
        # Layer 1: Inherited from 2-layer BNN (784 -> d)
        self.fc1 = BinarizedLinear(28 * 28, d, bias=False)
        
        # Layer 2: The Discrete Channel Layer
        self.channel = DiscreteChannelLayer(use_stochastic_channel=use_stochastic_channel)
        
        # Layer 3: Output layer (d -> 10)
        # Left as a standard linear layer for stable classification logits
        self.fc3 = nn.Linear(d, 10) 

    def forward(self, x):
        # Flatten MNIST image
        x = x.view(-1, 28 * 28)
        
        # # Convert to discrete integers {0, 1, ..., 255}
        # # We multiply by 255 and round to ensure they are discrete whole numbers
        # # We then cast to float so the Linear layer can process them mathematically
        # x = torch.round(x * 255.0)
        # Convert to strict bipolar {-1, 1}
        # > 0.5 becomes 1 (strokes), <= 0.5 becomes -1 (background)
        x = torch.where(x > 0.5, torch.tensor(1.0, device=x.device), torch.tensor(-1.0, device=x.device))
        
        # Pass through Binarized layer 1
        x = self.fc1(x)
        # apply relu for a fair comparison with no channel case
        x = F.relu(x)
        
        # Pass through the stochastic discrete channel
        # (Quantization layer removed to apply your idea to the wider range of integers)
        x = self.channel(x)
        
        # Final classification mapping
        x = self.fc3(x)
        return x

# ---------------------------------------------------------
# 4. Training and Testing Loops
# ---------------------------------------------------------
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Variables to track epoch-level metrics
    running_loss = 0.0
    correct = 0
    total = 0
    target_logs = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        # Track loss and accuracy
        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)

        # Log the positivity ratio from the channel layer
        target_logs.append(model.channel.last_target_ratio)
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    avg_positivity = sum(target_logs) / len(target_logs)
    
    print(f'Epoch: {epoch} | Average Loss: {epoch_loss:.4f} | Accuracy: {correct}/{total} ({epoch_acc:.2f}%) | Avg Target Ratio: {avg_positivity:.2%}')

    return epoch_loss, epoch_acc, avg_positivity

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'==> Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

def visualize_weights(model, epoch, save_path):
    """Visualizes the first 16 filters of the first binarized layer."""
    weights = BinarizeSTE.apply(model.fc1.weight).detach().cpu().numpy() # [d, 784]
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        # Reshape 784 to 28x28
        w_img = weights[i].reshape(28, 28)
        plt.imshow(w_img, cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Binarized Weights Filters (Epoch {epoch})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 128
    epochs = 40
    hidden_dim_d = 64
    learning_rate = 0.002

    # Device selection: Priority CUDA -> MPS (Apple Silicon) -> CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Use the DATASET environment variable if it exists, otherwise default to '../data'
    data_root = os.environ.get('DATASET', './data')
    print(f"Loading data from: {data_root}")
    # Download dataset locally
    train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_root, train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Set to False to disable the stochastic channel for ablation
    use_stochastic_channel = True

    # Initialize model and optimizer
    model = MNISTChannelNet(d=hidden_dim_d, use_stochastic_channel=use_stochastic_channel).to(device)
    # Adam optimizer usually works well with the Straight-Through Estimator
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Add a Learning Rate Scheduler: reduces LR by 50% every 5 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Lists to store metrics for plotting
    train_losses, train_accuracies, test_losses, test_accuracies, positivity_ratios = [], [], [], [], []
    
    # Training loop
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_target = train(model, device, train_loader, optimizer, epoch)
        te_loss, te_acc = test(model, device, test_loader)

        # Store metrics
        train_losses.append(tr_loss)
        train_accuracies.append(tr_acc)
        test_losses.append(te_loss)
        test_accuracies.append(te_acc)
        positivity_ratios.append(tr_target)

        # Visualize filters every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            visualize_weights(model, epoch, f'weights_epoch_{epoch}.png')

        scheduler.step()
        print(f"Learning Rate for next epoch: {scheduler.get_last_lr()[0]:.6f}\n")

    # Plotting Final Metrics
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    ax1.plot(range(1, epochs + 1), train_losses, label='Train')
    ax1.plot(range(1, epochs + 1), test_losses, label='Test')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(range(1, epochs + 1), train_accuracies, label='Train')
    ax2.plot(range(1, epochs + 1), test_accuracies, label='Test')
    ax2.set_title('Accuracy (%)')
    ax2.legend()

    ax3.plot(range(1, epochs + 1), positivity_ratios, color='green', label='Target Ratio')
    ax3.set_title('Feature Target Ratio (Pre-Channel)')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)