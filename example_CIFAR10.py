import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# 1. Binarization Operation with Straight-Through Estimator (STE)
# ---------------------------------------------------------------------------
class BinarizeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = input.sign()
        out[out == 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

binarize = BinarizeAct.apply

# ---------------------------------------------------------------------------
# 2. Binary Layers (Linear and Convolutional)
# ---------------------------------------------------------------------------
class BinaryLinear(nn.Linear):
    def forward(self, input):
        binary_weight = binarize(self.weight)
        return F.linear(input, binary_weight, self.bias)

class BinaryConv2d(nn.Conv2d):
    """
    A Convolutional layer where the weights are binarized during the forward pass.
    """
    def forward(self, input):
        binary_weight = binarize(self.weight)
        return F.conv2d(input, binary_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# ---------------------------------------------------------------------------
# 3. Model Architecture (Binary CNN)
# ---------------------------------------------------------------------------
class BNN_CNN(nn.Module):
    """
    A simple VGG-style Convolutional Neural Network for CIFAR-10 using Binary weights.
    Architecture:
    2x (Conv -> BN -> Binarize) -> MaxPool -> 2x (Conv -> BN -> Binarize) -> MaxPool -> FC -> FC
    """
    def __init__(self):
        super(BNN_CNN, self).__init__()
        
        # Block 1
        self.conv1 = BinaryConv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = BinaryConv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces 32x32 to 16x16
        
        # Block 2
        self.conv3 = BinaryConv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = BinaryConv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces 16x16 to 8x8

        # Fully Connected Classifier
        self.fc1 = BinaryLinear(256 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = BinaryLinear(512, 10)
        self.bn6 = nn.BatchNorm1d(10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = binarize(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool1(x)
        x = binarize(x)

        # Block 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = binarize(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool2(x)
        x = binarize(x)

        # Flatten for Dense Layers
        x = x.view(-1, 256 * 8 * 8)
        
        # Classifier
        x = self.fc1(x)
        x = self.bn5(x)
        x = binarize(x)
        
        # Output layer (no binarization before CrossEntropyLoss)
        x = self.fc2(x)
        x = self.bn6(x)
        
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
        
        # Clamp weights between -1 and 1
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
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')

# ---------------------------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------------------------
def main():
    batch_size = 128
    epochs = 10  # CIFAR-10 usually requires more epochs to converge
    
    # Priority CUDA -> MPS (Apple Silicon) -> CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = os.environ.get('DATASET', '../data')
    print(f"Loading data from: {data_root}")

    # CIFAR-10 Transforms (Adding basic data augmentation for the training set)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 3 channels for RGB
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10
    dataset1 = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
    dataset2 = datasets.CIFAR10(data_root, train=False, transform=test_transform)
    
    train_loader = DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=False, num_workers=2)

    model = BNN_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005) # Slightly lower LR for CNN stability

    print("\nStarting Training on CIFAR-10 (Binary weights & Binary activations)")
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()