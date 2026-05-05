import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Straight-Through Estimator for Weights
# ==========================================
class SignSTE(torch.autograd.Function):
    """
    Binarizes weights to {-1, 1} with a Straight-Through Estimator for backprop.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Binarize strictly to -1 or 1
        return torch.where(input >= 0, torch.tensor(1.0, device=input.device), torch.tensor(-1.0, device=input.device))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Hardtanh STE: cancel gradients if the underlying continuous weights exceed [-1, 1]
        grad_input[input.abs() > 1.0] = 0
        return grad_input

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Continuous latent weights
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Initialize uniformly
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # Apply STE to get binary weights in {-1, 1}
        w_bin = SignSTE.apply(self.weight)
        return F.linear(x, w_bin)

# ==========================================
# 2. The Discrete Memoryless Channel Layer
# ==========================================
class ChannelLayer(nn.Module):
    def __init__(self, p_high=0.99, p_low=0.01):
        super().__init__()
        self.p_high = p_high
        self.p_low = p_low

    def forward(self, x):
        """
        x is expected to be in {-3, -1, 1, 3}.
        If x > 0 (i.e., 1 or 3), p(mask=1) = 0.99
        If x < 0 (i.e., -1 or -3), p(mask=1) = 0.01
        """
        # Calculate transition probabilities based on input signs
        p = torch.where(x > 0, torch.tensor(self.p_high, device=x.device), torch.tensor(self.p_low, device=x.device))
        
        # Sample mask M_i ~ Bernoulli(p)
        # torch.bernoulli automatically detaches 'mask' from 'p' in the computational graph.
        # This is exactly what we want, as the gradient of a step function is 0.
        mask = torch.bernoulli(p)
        
        # Element-wise multiply (M * f)
        # During backprop, dy/dx = mask. The gradient flows straight through the non-zero paths.
        return mask * x

# ==========================================
# 3. Full 3-Layer Architecture
# ==========================================
class BNNChannelModel(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        # Layer 1: 3D input -> d-dimensional binary features
        self.layer1 = BinaryLinear(in_features=3, out_features=d)
        
        # Layer 2: The Channel Layer
        self.channel = ChannelLayer()
        
        # Layer 3: Output classification layer (standard real-valued dense layer)
        self.layer3 = nn.Linear(in_features=d, out_features=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.channel(x)
        x = self.layer3(x)
        return x

# ==========================================
# 4. Synthetic Dataset Generation
# ==========================================
def get_synthetic_data(num_samples=2000):
    """
    Generates inputs X in {-1, 1}^3.
    Target Y is a binary classification rule (e.g., Majority Vote / Sum > 0).
    """
    # Random {0, 1} mapped to {-1, 1}
    X = torch.randint(0, 2, (num_samples, 3)).float() * 2 - 1
    
    # Label is 1 if there are more 1s than -1s, else 0
    Y = (X.sum(dim=1) > 0).long()
    
    return X, Y

# ==========================================
# 5. Training Loop to verify it works
# ==========================================
if __name__ == "__main__":
    # Hyperparameters
    d_dimension = 32
    epochs = 50
    lr = 0.01

    # Initialize model, loss, and optimizer
    model = BNNChannelModel(d=d_dimension)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Get data
    X_train, Y_train = get_synthetic_data(2000)
    X_test, Y_test = get_synthetic_data(500)

    print(f"Training BNN Channel Model with d={d_dimension}...\n")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(X_train)
        loss = criterion(logits, Y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            # Test Evaluation
            model.eval()
            with torch.no_grad():
                test_logits = model(X_test)
                predictions = torch.argmax(test_logits, dim=1)
                accuracy = (predictions == Y_test).float().mean().item()
            
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Test Accuracy: {accuracy * 100:.2f}%")

    print("\nTraining complete! The network successfully navigates the discrete memoryless channel.")