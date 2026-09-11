"""Few-shot Flower Classification project
Implements baseline, Siamese, and Prototypical networks.
Run with:
 python train_few_shot.py --mode baseline
 python train_few_shot.py --mode siamese
 python train_few_shot.py --mode proto
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import wandb

# ---------------- Data utilities ----------------

def get_loader(data_dir, split='train', batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    folder = ImageFolder(root=os.path.join(data_dir, split), transform=transform)
    loader = DataLoader(folder, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
    return loader, folder.classes

# ---------------- Models ----------------

class BaselineModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.backbone = backbone
    def forward_one(self, x):
        return self.backbone(x)
    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)

class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.backbone = backbone
    def embed(self, x):
        return self.backbone(x)

# ---------------- Losses ----------------

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, out1, out2, label):
        # label=1 same, 0 different
        d = F.pairwise_distance(out1, out2)
        loss = torch.mean((1-label) * d.pow(2) +
                          label * F.relu(self.margin - d).pow(2))
        return loss

# ---------------- Training utils ----------------

def train_baseline(model, loader, optimizer, criterion, device):
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
    return total_loss/total, correct/total

def evaluate_baseline(model, loader, criterion, device):
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
    return total_loss/total, correct/total

# Episodic sampler for proto/siamese

def create_episode(dataset, n_way=5, k_shot=5, q_query=15):
    # Map class -> indices
    class_idx = {c: [] for c in range(len(dataset.classes))}
    for i, (_, label) in enumerate(dataset.samples):
        class_idx[label].append(i)
    chosen_classes = np.random.choice(list(class_idx.keys()), n_way, replace=False)
    support_idxs, query_idxs, support_labels, query_labels = [], [], [], []
    for cls_id, cls in enumerate(chosen_classes):
        idxs = np.random.choice(class_idx[cls], k_shot + q_query, replace=False)
        support = idxs[:k_shot]
        query = idxs[k_shot:]
        support_idxs.extend(support)
        query_idxs.extend(query)
        support_labels.extend([cls_id]*k_shot)
        query_labels.extend([cls_id]*q_query)
    support_set = Subset(dataset, support_idxs)
    query_set = Subset(dataset, query_idxs)
    support_loader = DataLoader(support_set, batch_size=len(support_idxs), shuffle=False)
    query_loader = DataLoader(query_set, batch_size=len(query_idxs), shuffle=False)
    return support_loader, query_loader, support_labels, query_labels

def train_prototypical(model, dataset, optimizer, device, episodes=200, n_way=5, k_shot=5, q_query=15):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    for ep in range(episodes):
        sup_loader, qry_loader, sup_labels, qry_labels = create_episode(dataset, n_way, k_shot, q_query)
        xs, _ = next(iter(sup_loader))
        xq, _ = next(iter(qry_loader))
        xs, xq = xs.to(device), xq.to(device)
        zs = model.embed(xs).view(n_way, k_shot, -1).mean(1)  # prototypes
        zq = model.embed(xq)
        dists = torch.cdist(zq, zs)  # (Nq, n_way)
        logits = -dists
        loss = ce(logits, torch.tensor(qry_labels, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/episodes

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Few‑shot flower classification")
    parser.add_argument("--mode", choices=["baseline", "siamese", "proto"], required=True)
    parser.add_argument("--data_dir", default="data/flowers")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shots", type=int, default=5, help="k‑shot for proto/siamese")
    parser.add_argument("--ways", type=int, default=5, help="n‑way for proto/siamese")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project='few-shot-flower', config=vars(args))

    if args.mode == "baseline":
        train_loader, class_names = get_loader(args.data_dir, 'train', args.batch_size, shuffle=True)
        val_loader, _ = get_loader(args.data_dir, 'val', args.batch_size, shuffle=False)
        model = BaselineModel(num_classes=len(class_names)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1, args.epochs+1):
            tr_loss, tr_acc = train_baseline(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate_baseline(model, val_loader, criterion, device)
            wandb.log({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                       "val_loss": val_loss, "val_acc": val_acc})
            print(f"Epoch {epoch}: Train {tr_acc:.3f}, Val {val_acc:.3f}")
    elif args.mode == "siamese":
        # Build dataset for pairs
        base_dataset = ImageFolder(root=os.path.join(args.data_dir, 'train'),
                                   transform=transforms.Compose([
                                       transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485,0.456,0.406],
                                                            std=[0.229,0.224,0.225])]))
        model = SiameseNetwork(embedding_dim=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = ContrastiveLoss(margin=1.0)
        for epoch in range(1, args.epochs+1):
            model.train()
            epoch_loss = 0.0
            for _ in range(100):
                # positive pair
                cls = np.random.choice(len(base_dataset.classes))
                idxs = [i for i, (_, l) in enumerate(base_dataset.samples) if l == cls]
                i1, i2 = np.random.choice(idxs, 2, replace=False)
                img1, _ = base_dataset[i1]
                img2, _ = base_dataset[i2]
                # negative pair
                neg_cls = (cls + np.random.randint(1, len(base_dataset.classes))) % len(base_dataset.classes)
                neg_idxs = [i for i, (_, l) in enumerate(base_dataset.samples) if l == neg_cls]
                i3 = np.random.choice(neg_idxs)
                img3, _ = base_dataset[i3]
                # convert to batch
                pos1 = img1.unsqueeze(0).to(device)
                pos2 = img2.unsqueeze(0).to(device)
                neg1 = img1.unsqueeze(0).to(device)
                neg2 = img3.unsqueeze(0).to(device)
                out_pos1, out_pos2 = model(pos1, pos2)
                out_neg1, out_neg2 = model(neg1, neg2)
                loss_pos = criterion(out_pos1, out_pos2, torch.tensor([1.0], device=device))
                loss_neg = criterion(out_neg1, out_neg2, torch.tensor([0.0], device=device))
                loss = loss_pos + loss_neg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            wandb.log({"epoch": epoch, "loss": epoch_loss/100})
            print(f"Epoch {epoch} loss {epoch_loss/100:.4f}")
    else:  # proto
        base_dataset = ImageFolder(root=os.path.join(args.data_dir, 'train'),
                                   transform=transforms.Compose([
                                       transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485,0.456,0.406],
                                                            std=[0.229,0.224,0.225])]))
        model = PrototypicalNetwork(embedding_dim=256).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        avg_loss = train_prototypical(model, base_dataset, optimizer, device,
                                      episodes=args.epochs, n_way=args.ways,
                                      k_shot=args.shots, q_query=args.shots)
        wandb.log({"proto_loss": avg_loss})
        print(f"ProtoNet avg loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
