import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────
# STEP 1: Check PyTorch
# ─────────────────────────────────────────
print("=" * 55)
print("STEP 1: Checking Setup...")
print("=" * 55)
print(f"PyTorch version : {torch.__version__}")
print(f"Using device    : {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print()

# ─────────────────────────────────────────
# STEP 2: Load MNIST Dataset
# ─────────────────────────────────────────
print("=" * 55)
print("STEP 2: Loading MNIST Dataset...")
print("=" * 55)

transform = transforms.Compose([
    transforms.ToTensor(),                        # converts image to tensor
    transforms.Normalize((0.5,), (0.5,))          # normalize to [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print(f"Training images : {len(train_dataset)}")
print(f"Testing images  : {len(test_dataset)}")
print(f"Batch size      : 64\n")

# ─────────────────────────────────────────
# STEP 3: Visualize Sample Digits
# ─────────────────────────────────────────
print("=" * 55)
print("STEP 3: Saving Sample Digit Images...")
print("=" * 55)

images, labels = next(iter(train_loader))

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sample Digits from MNIST Dataset", fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    img = images[i].squeeze().numpy()
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Label: {labels[i].item()}", fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.savefig("sample_digits.png", dpi=100)
plt.close()
print("Saved → sample_digits.png\n")

# ─────────────────────────────────────────
# STEP 4: Build CNN Model
# ─────────────────────────────────────────
print("=" * 55)
print("STEP 4: Building CNN Model...")
print("=" * 55)

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()

        # Block 1: Conv → ReLU → MaxPool
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # 1 input channel (grayscale), 32 filters
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Block 2: Conv → ReLU → MaxPool
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 32 → 64 filters
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),        # 64 filters × 5×5 feature map
            nn.ReLU(),
            nn.Dropout(0.3),                   # prevent overfitting
            nn.Linear(128, 10)                 # 10 output classes (digits 0-9)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc(x)
        return x

model     = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)
print()

# ─────────────────────────────────────────
# STEP 5: Train the Model
# ─────────────────────────────────────────
print("=" * 55)
print("STEP 5: Training the Model (5 epochs)...")
print("=" * 55)

EPOCHS = 5
train_accuracies = []
train_losses     = []

for epoch in range(EPOCHS):
    model.train()
    running_loss    = 0.0
    correct         = 0
    total           = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()           # clear previous gradients
        outputs = model(images)         # forward pass
        loss    = criterion(outputs, labels)  # compute loss
        loss.backward()                 # backward pass
        optimizer.step()                # update weights

        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        correct      += (predicted == labels).sum().item()
        total        += labels.size(0)

    acc  = correct / total * 100
    loss_avg = running_loss / len(train_loader)
    train_accuracies.append(acc)
    train_losses.append(loss_avg)

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {loss_avg:.4f}  Accuracy: {acc:.2f}%")

print()

# ─────────────────────────────────────────
# STEP 6: Evaluate on Test Data
# ─────────────────────────────────────────
print("=" * 55)
print("STEP 6: Evaluating on Test Data...")
print("=" * 55)

model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs        = model(images)
        _, predicted   = torch.max(outputs, 1)
        correct       += (predicted == labels).sum().item()
        total         += labels.size(0)

test_acc = correct / total * 100
print(f"Test Accuracy : {test_acc:.2f}%\n")

# ─────────────────────────────────────────
# STEP 7: Plot Training History
# ─────────────────────────────────────────
print("=" * 55)
print("STEP 7: Saving Training Graph...")
print("=" * 55)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Training History", fontsize=14, fontweight='bold')

ax1.plot(range(1, EPOCHS+1), train_accuracies, marker='o', color='blue')
ax1.set_title('Accuracy per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.grid(True)

ax2.plot(range(1, EPOCHS+1), train_losses, marker='o', color='red')
ax2.set_title('Loss per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=100)
plt.close()
print("Saved → training_history.png\n")

# ─────────────────────────────────────────
# STEP 8: Predict Sample Test Images
# ─────────────────────────────────────────
print("=" * 55)
print("STEP 8: Predicting Sample Test Images...")
print("=" * 55)

test_images, test_labels = next(iter(test_loader))
test_images = test_images.to(device)

model.eval()
with torch.no_grad():
    outputs     = model(test_images)
    _, predicted = torch.max(outputs, 1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Model Predictions vs Actual Labels", fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    img   = test_images[i].cpu().squeeze().numpy()
    pred  = predicted[i].cpu().item()
    actual = test_labels[i].item()
    color = 'green' if pred == actual else 'red'
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Pred: {pred} | Real: {actual}", color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig("predictions.png", dpi=100)
plt.close()
print("Saved → predictions.png\n")

# ─────────────────────────────────────────
# STEP 9: Save the Model
# ─────────────────────────────────────────
print("=" * 55)
print("STEP 9: Saving Model...")
print("=" * 55)

torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved → mnist_model.pth\n")

print("=" * 55)
print(f"✅ All done! Final Test Accuracy: {test_acc:.2f}%")
print("=" * 55)
print("\nFiles created:")
print("  📊 sample_digits.png     — sample images from dataset")
print("  📈 training_history.png  — accuracy & loss graphs")
print("  🔍 predictions.png       — model predictions vs actual")
print("  💾 mnist_model.pth       — saved trained model")
