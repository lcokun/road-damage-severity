from __future__ import annotations

from typing import Tuple
import numpy as np
import torch


def train_one_epoch(model, loader, optimizer, criterion, device: str) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device: str) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    correct, total = 0, 0
    ys, preds = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        p = logits.argmax(dim=1)

        correct += (p == y).sum().item()
        total += y.size(0)

        ys.append(y.cpu().numpy())
        preds.append(p.cpu().numpy())

    acc = correct / max(total, 1)
    return acc, np.concatenate(ys), np.concatenate(preds)