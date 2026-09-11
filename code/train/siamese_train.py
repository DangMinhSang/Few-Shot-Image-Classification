import argparse
import torch
import torch.nn as nn
import wandb
import numpy as np
from ..models.siamese import SiameseNet
from ..losses.contrastive import ContrastiveLoss
from ..data.data_utils import set_seed, get_image_folder, create_episode
from ..config import SEED, LR, BATCH_SIZE

def train_one_epoch(model, dataset, optimizer, criterion, device, n_way=5, k_shot=5, q_query=15):
    model.train()
    epoch_loss = 0.0
    for _ in range(100):  # number of random episodes per epoch
        support_loader, query_loader, s_labels, q_labels = create_episode(dataset, n_way, k_shot, q_query)
        # Positive pair from support set (same class)
        # Choose a random class from support set
        cls = np.random.choice(len(dataset.classes))
        idxs = [i for i, (_, l) in enumerate(dataset.samples) if l == cls]
        i1, i2 = np.random.choice(idxs, 2, replace=False)
        img1, _ = dataset[i1]
        img2, _ = dataset[i2]
        # Negative pair (different class)
        neg_cls = (cls + np.random.randint(1, len(dataset.classes))) % len(dataset.classes)
        neg_idxs = [i for i, (_, l) in enumerate(dataset.samples) if l == neg_cls]
        i3 = np.random.choice(neg_idxs)
        img3, _ = dataset[i3]
        # Prepare batches
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
    return epoch_loss / 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--ways", type=int, default=5)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="few-shot-flower", config=vars(args))

    # Use training split for episode generation
    train_dataset, _ = get_image_folder(split="train")

    model = SiameseNet(embedding_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = ContrastiveLoss(margin=1.0)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_dataset, optimizer, criterion, device,
                                 n_way=args.ways, k_shot=args.shots, q_query=args.shots)
        wandb.log({"epoch": epoch, "loss": loss})
        print(f"Epoch {epoch:02d} loss {loss:.4f}")

if __name__ == "__main__":
    main()
