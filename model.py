from datetime import timedelta
from pathlib import Path
from time import perf_counter

import torch
from torch import nn, optim
from torchvision.models import resnet18

from data import make_data_loader

checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / "best_resnet18.pt"

TRAIN_SEED = 42
epochs = 200

def make_model():
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()

    return model

def train_model(model):
    train_loader, val_loader, _, _ = make_data_loader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 장치: {device}", flush=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_accuracy = -1.0
    training_start = perf_counter()

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_total += batch_size

        train_loss = train_loss_sum / train_total
        model.eval()

        loss_sum = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                logits = model(images)
                batch_loss = criterion(logits, labels)
                batch_size = labels.size(0)
                loss_sum += batch_loss.item() * batch_size
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += batch_size

        val_loss = loss_sum / total
        val_accuracy = 100 * correct / total

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "seed": TRAIN_SEED,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_accuracy,
            }, checkpoint_path)

        scheduler.step()
        elapsed = perf_counter() - training_start
        remaining = elapsed / (epoch + 1) * (epochs - epoch - 1)
        eta = timedelta(seconds=round(remaining))
        print(
            f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.2f}%, "
            f"예상 남은 시간={eta}",
            flush=True,
        )


def main():
    torch.manual_seed(TRAIN_SEED)
    print(f"학습 seed: {TRAIN_SEED}", flush=True)
    train_model(make_model())

if __name__ == "__main__":
    main()
