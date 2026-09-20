"""
train_model.py
===============
Run this ONCE, locally, to train a 3-layer neural network (Input -> Hidden ->
Output) on the MNIST dataset using PyTorch, and save it as `mnist_pytorch.pt`.
app.py will automatically pick it up on its next run.

Usage:
    python train_model.py

This downloads the MNIST dataset via torchvision (~11MB, needs an internet
connection the first time) and trains for several epochs — a couple of
minutes on a normal laptop CPU.

Honest note on accuracy: a well-trained network like this typically reaches
around 97-98% accuracy on the standard MNIST test set. No handwritten-digit
model can honestly promise literal 100% accuracy on arbitrary new drawings —
messy or ambiguous strokes will occasionally confuse any classifier.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DigitClassifier(nn.Module):
    """A classic 3-layer feedforward neural network:
       Input Layer (784 neurons) -> Hidden Layer (128 neurons) -> Output Layer (10 neurons)
    This is the exact architecture visualized in the app's network diagram."""

    def __init__(self):
        super().__init__()
        self.input_to_hidden = nn.Linear(28 * 28, 128)   # Input layer -> Hidden layer
        self.hidden_to_output = nn.Linear(128, 10)         # Hidden layer -> Output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)                       # Flatten 28x28 image into a 784-length vector (tensor op)
        hidden = self.relu(self.input_to_hidden(x))     # Hidden layer activation
        logits = self.hidden_to_output(hidden)           # Output layer (raw scores per digit)
        return logits


def main():
    print("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="mnist_data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="mnist_data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = DigitClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 12
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - loss: {total_loss / len(train_loader):.4f}")

    # Evaluate on the held-out test set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
    print("(Real-world drawings may score a bit lower than the clean test set above.)")

    torch.save(model.state_dict(), "mnist_pytorch.pt")
    print("Saved model to mnist_pytorch.pt — you can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
