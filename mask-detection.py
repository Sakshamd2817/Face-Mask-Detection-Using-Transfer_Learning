import os
import shutil
import random
from torch.utils.data import DataLoader, TensorDataset , random_split
from torchvision import datasets , transforms , models
import torch
import torch.nn as nn

source_dir = "dataset"      # has mask / no_mask
target_dir = "data_split"  # will be created

classes = ["with_mask", "without_mask"]

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

for cls in classes:
    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, imgs in splits.items():
        split_dir = os.path.join(target_dir, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for img in imgs:
            src = os.path.join(source_dir, cls, img)
            dst = os.path.join(split_dir, img)
            shutil.copy(src, dst)

folder = "data_split"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.88,0.98,0.1],
        std=[0.01,0.22,0.43]
    )

]
)

train_data = datasets.ImageFolder("data_split/train", transform=transform)
val_data   = datasets.ImageFolder("data_split/val", transform=transform)
test_data  = datasets.ImageFolder("data_split/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_acc = 100 * correct / total
print(f"Validation Accuracy: {val_acc:.2f}%")
