import argparse
import torch
import torch.nn as nn
import wandb
import numpy as np
from ..models.prototypical import ProtoNet
from ..data.data_utils import set_seed, get_image_folder, create_episode
from ..config import SEED, LR, BATCH_SIZE

def train_prototypical(model, dataset, optimizer, device, episodes=200, n_way=5, k_shot=5, q_query=15):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    for ep in range(episodes):
        sup_loader, qry_loader, sup_labels, qry_labels = create_episode(dataset, n_way, k_shot, q_query)
        xs, _ = next(iter(sup_loader))
        xq, _ = next(iter(qry_loader))
        xs, xq = xs.to(device), xq.to(device)
        zs = model.embed(xs).view(n_way, k_shot, -1).mean(1)   # prototypes
        zq = model.embed(xq)
        dists = torch.cdist(zq, zs)  # (Nq, n_way)
        logits = -dists
        loss = ce(logits, torch.tensor(qry_labels, device=device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / episodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of episodes")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--ways", type=int, default=5)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="few-shot-flower", config=vars(args))

    train_dataset, _ = get_image_folder(split="train")
    model = ProtoNet(embedding_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    avg_loss = train_prototypical(model, train_dataset, optimizer, device,
                                 episodes=args.epochs, n_way=args.ways,
                                 k_shot=args.shots, q_query=args.shots)
    wandb.log({"proto_avg_loss": avg_loss})
    print(f"ProtoNet avg loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
