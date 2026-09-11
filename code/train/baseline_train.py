import argparse
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from ..models.baseline import BaselineResNet
from ..data.data_utils import set_seed, get_image_folder
from ..config import SEED, LR, BATCH_SIZE

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, pred = torch.max(logits, 1)
        correct += (pred == targets).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * imgs.size(0)
            _, pred = torch.max(logits, 1)
            correct += (pred == targets).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="few-shot-flower", config=vars(args))

    train_loader, classes = get_image_folder(split="train")
    val_loader, _ = get_image_folder(split="val")

    model = BaselineResNet(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        wandb.log({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                   "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch:02d} | Train {tr_acc:.3f} | Val {val_acc:.3f}")

if __name__ == "__main__":
    main()
