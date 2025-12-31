import os
from typing import Dict, Tuple

import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor
from torchvision.models import VisionTransformer
from tqdm import tqdm


def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Transform to normalize the input images
    transform = transforms.Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    with FileLock(os.path.expanduser("~/data.lock")):
        # Download training data from open datasets
        training_data = datasets.CIFAR10(
            root="~/data",
            train=True,
            download=True,
            transform=transform,
        )

        # Download test data from open datasets
        testing_data = datasets.CIFAR10(
            root="~/data",
            train=False,
            download=True,
            transform=transform,
        )

    # Create data loaders,
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def train_func():
    lr = 1e-3
    epochs = 1
    batch_size = 512

    # Get data loaders inside the worke training function.
    train_dataloader, valid_dataloader = get_dataloaders(batch_size=batch_size)

    model = VisionTransformer(
        image_size=32,  # CIFAR-10 images are 32x32
        patch_size=4,   # Patch size of 4
        num_layers=12,  # Number of transformer layers
        num_heads=8,    # Number of attention heads
        hidden_dim=384, # Hidden size (can be adjusted)
        mlp_dim=768,    # MLP dimension (can be adjusted)
        num_classes=10, # CIFAR-10 has 10 classes
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    # Model training loop
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        valid_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(valid_dataloader, desc=f"Valid Epoch {epoch}"):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)

                valid_loss += loss.item()
                num_total += y.shape[0]
                num_correct += (pred.argmax(1) == y).sum().item()

        valid_loss /= len(train_dataloader)
        accuracy = num_correct / num_total

        print({"epoch_num": epoch, "loss": valid_loss, "accuracy": accuracy})



def main():
    print("Hello from ray-tutorial!")
    train_func()


if __name__ == "__main__":
    main()
