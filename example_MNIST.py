import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# 1. Binarization Operation with Straight-Through Estimator (STE)
# ---------------------------------------------------------------------------
class BinarizeAct(torch.autograd.Function):
    """
    Custom autograd function for binarizing activations and weights.
    Forward pass: applies the sign function (output is +1 or -1).
    Backward pass: uses the Straight-Through Estimator (STE) which acts like a HardTanh to allow gradients to flow back, ignoring the non-differentiable step.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Binarize to -1 or +1. If exactly 0, default to 1.
        out = input.sign()
        out[out == 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # STE: cancel the gradient if the input is outside the [-1, 1] range.
        # This prevents weights from growing infinitely.
        grad_input[input.abs() > 1] = 0
        return grad_input

# Alias to make it easy to use
binarize = BinarizeAct.apply


# ---------------------------------------------------------------------------
# 2. Binary Linear Layer
# ---------------------------------------------------------------------------
class BinaryLinear(nn.Linear):
    """
    A Linear layer where the weights are binarized during the forward pass.
    The underlying stored weights remain full-precision floats to accumulate 
    tiny gradient updates.
    """
    def forward(self, input):
        # Binarize the weights before applying the linear transformation
        binary_weight = binarize(self.weight)

        # try binary bias
        binary_bias = binarize(self.bias) if self.bias is not None else None
        
        # We generally do not binarize the bias, though some extreme BNNs do.
        return F.linear(input, binary_weight, binary_bias)

# ---------------------------------------------------------------------------
# 3. Model Architecture
# ---------------------------------------------------------------------------
class BNN_MLP(nn.Module):
    """
    A Multi-Layer Perceptron using Binary weights and Binary activations.
    Architecture:
    Input -> [BinaryLinear -> BatchNorm -> Binarize] x2 -> BinaryLinear -> BatchNorm -> Output
    """
    def __init__(self):
        super(BNN_MLP, self).__init__()
        # Flattened 28x28 MNIST image = 784
        self.fc1 = BinaryLinear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = BinaryLinear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        # Output layer for 10 classes
        self.fc3 = BinaryLinear(512, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        # Flatten image (Batch Size, 1, 28, 28) -> (Batch Size, 784)
        x = x.view(-1, 28 * 28)
        
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = binarize(x) # Binarize activations
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = binarize(x) # Binarize activations
        
        # Layer 3 (Output layer: do not binarize the final output before CrossEntropy)
        x = self.fc3(x)
        x = self.bn3(x)
        
        return x

# ---------------------------------------------------------------------------
# 4. Training and Testing Loop
# ---------------------------------------------------------------------------
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        optimizer.step()
        
        # Clamp the real-valued weights between -1 and 1 after the update.
        # This is a standard BNN trick to ensure weights don't grow too large 
        # and stay close to their binarized representations.
        for p in model.parameters():
            p.data.clamp_(-1, 1)
            
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)      # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')
    
def verify_binarity(model, device, test_loader):
    """
    Demonstrates that weights and activations are indeed binary (-1 or 1).
    """
    print("\n" + "="*50)
    print("VERIFYING BINARITY")
    print("="*50)
    
    model.eval()
    
    # 1. Inspect Weights of the first layer
    # We manually binarize because the layer stores floats but uses binary in forward()
    with torch.no_grad():
        raw_weights = model.fc1.weight
        binary_weights = binarize(raw_weights)
        
        unique_values = torch.unique(binary_weights)
        print(f"Unique values in binarized fc1 weights: {unique_values.tolist()}")
        print(f"Sample of 5x5 binary weight matrix:\n{binary_weights[:5, :5]}")

    # 2. Inspect Activations (Features) of the first hidden layer
    # We'll take one batch and manually run the first part of the forward pass
    data, _ = next(iter(test_loader))
    data = data.to(device)
    
    with torch.no_grad():
        x = data.view(-1, 28 * 28)
        x = model.fc1(x)
        x = model.bn1(x)
        features = binarize(x) # These are the features passed to layer 2
        
        unique_features = torch.unique(features)
        print(f"\nUnique values in fc1 output features: {unique_features.tolist()}")
        print(f"Sample of features for the first image (first 20 neurons):\n{features[0, :20]}")
    
    print("="*50 + "\n")

# ---------------------------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------------------------
def main():
    # Setup
    batch_size = 128
    epochs = 10
    # Device selection: Priority CUDA -> MPS (Apple Silicon) -> CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms (Normalize inputs to range roughly [-1, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    # Use the DATASET environment variable if it exists, otherwise default to '../data'
    data_root = os.environ.get('DATASET', './data')
    print(f"Loading data from: {data_root}")
    # Datasets & Loaders
    dataset1 = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(data_root, train=False, transform=transform)
    
    train_loader = DataLoader(dataset1, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=False)

    # Initialize model and optimizer
    model = BNN_MLP().to(device)
    
    # Adam usually works well with BNNs. High learning rates are common to escape zero-gradients.
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("\nStarting Training (Binary weights & Binary activations)")
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Run the verification at the end
    verify_binarity(model, device, test_loader)

if __name__ == '__main__':
    main()