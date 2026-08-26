# src/train_directml.py
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch_directml

DATA_DIR = "data"
MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Detect AMD Radeon GPU via DirectML
device = torch_directml.device()
print(f"[GPU ACCELERATION]: Using AMD Radeon GPU via DirectML ({device})")

# Preprocessing & Data Augmentations
train_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloaders(batch_size=32):
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    return train_loader, test_loader, class_names

def train_model(model_name="resnet50", epochs=15, batch_size=32):
    train_loader, test_loader, class_names = get_dataloaders(batch_size=batch_size)
    num_classes = len(class_names)
    print(f"\n--- Training {model_name} on AMD Radeon GPU ({device}) ---")

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    else:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0
    save_path = os.path.join(MODELS_DIR, f"best_gpu_{model_name}.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        start_t = time.time()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == y_batch.data)
            total += x_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total

        # Evaluation on test set
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                out = model(x_b)
                _, p = torch.max(out, 1)
                val_correct += torch.sum(p == y_b.data)
                val_total += x_b.size(0)

        val_acc = (val_correct.double() / val_total).item()
        scheduler.step(val_acc)
        elapsed = time.time() - start_t

        print(f"Epoch {epoch+1}/{epochs} [{elapsed:.1f}s] - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | Test Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best GPU model ({best_acc*100:.2f}%) to {save_path}")

    print(f"\n[RESULT] Finished Training {model_name}. Best Test Accuracy: {best_acc*100:.2f}%\n")
    return best_acc

def main():
    train_model("resnet50", epochs=15)
    train_model("mobilenet", epochs=15)

if __name__ == "__main__":
    main()
