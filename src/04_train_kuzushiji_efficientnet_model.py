import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from efficientnet import EfficientNetFactory

# 1. Load numpy arrays
print("Loading data...")
images = np.load("../data/kuzushiji-arrays/images.npy")
labels = np.load("../data/kuzushiji-arrays/labels.npy")

# Convert to PyTorch tensors
images_tensor = torch.from_numpy(images).float().unsqueeze(1) / 255.0  # Add channel dimension and normalize
labels_tensor = torch.from_numpy(labels).long()

# 2. Split data into training and validation sets
print("Splitting data...")
dataset = TensorDataset(images_tensor, labels_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 3. Create PyTorch DataLoaders
print("Creating DataLoaders...")
batch_size = 8192
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 4. Set up EfficientNet model
print("Setting up model...")
num_classes = len(np.unique(labels))
model = EfficientNetFactory.create('b0', num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# 5. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

# 6. Training loop
print("Starting training...")
num_epochs = 50
log_file = "training_log.jsonl"

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_true = []
    train_pred = []

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

        train_true.extend(targets.cpu().numpy())
        train_pred.extend(predicted.cpu().numpy())

        # Checkpoint every 100 batches
        if (batch_idx + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pth')

    train_accuracy = train_correct / train_total
    train_precision = precision_score(train_true, train_pred, average='weighted')
    train_recall = recall_score(train_true, train_pred, average='weighted')

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_true = []
    val_pred = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

            val_true.extend(targets.cpu().numpy())
            val_pred.extend(predicted.cpu().numpy())

    val_accuracy = val_correct / val_total
    val_precision = precision_score(val_true, val_pred, average='weighted')
    val_recall = recall_score(val_true, val_pred, average='weighted')
    val_confusion_matrix = confusion_matrix(val_true, val_pred).tolist()

    # Log metrics
    log_entry = {
        "epoch": epoch + 1,
        "train_loss": train_loss / len(train_loader),
        "train_accuracy": train_accuracy,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "val_loss": val_loss / len(val_loader),
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_confusion_matrix": val_confusion_matrix
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {log_entry['train_loss']:.4f}, Train Acc: {train_accuracy:.4f}")
    print(f"Val Loss: {log_entry['val_loss']:.4f}, Val Acc: {val_accuracy:.4f}")

print("Training complete!")
