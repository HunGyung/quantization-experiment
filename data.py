from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


DATA_DIR = Path(__file__).resolve().parent / "data"
SPLIT_PATH = DATA_DIR / "split_indices.npz"


def split_data(raw_train):
    """CIFAR-10 원본 학습 데이터 50,000장의 인덱스를 클래스별로 나눈다."""
    rng = np.random.default_rng(42)
    labels = np.asarray(raw_train.targets)
    splits = {"train": [], "validation": [], "calibration": []}

    for cls in range(10):
        indices = np.flatnonzero(labels == cls)
        rng.shuffle(indices)

        splits["calibration"].extend(indices[:100])
        splits["validation"].extend(indices[100:600])
        splits["train"].extend(indices[600:])

    splits = {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in splits.items()
    }
    for indices in splits.values():
        rng.shuffle(indices)

    assert len(splits["train"]) == 44_000
    assert len(splits["validation"]) == 5_000
    assert len(splits["calibration"]) == 1_000
    assert len(set(splits["train"]) | set(splits["validation"]) |
            set(splits["calibration"])) == 50_000

    np.savez(SPLIT_PATH, **splits)


def load_or_create_splits(raw_train):
    if not SPLIT_PATH.exists():
        split_data(raw_train)

    with np.load(SPLIT_PATH) as saved:
        return saved["train"], saved["validation"], saved["calibration"]


def calculate_mean_std(raw_train, train_indices):
    """학습용 44,000장의 RGB 채널별 평균과 표준편차를 0~1 범위로 계산한다."""
    pixels = raw_train.data[train_indices]
    mean = pixels.mean(axis=(0, 1, 2), dtype=np.float64) / 255.0
    std = pixels.std(axis=(0, 1, 2), dtype=np.float64) / 255.0
    return mean.tolist(), std.tolist()


def make_datasets():
    # 인덱스를 만들고 정규화 값을 계산할 때는 변환되지 않은 원본을 사용한다.
    raw_train = CIFAR10(root=DATA_DIR, train=True, download=True)
    train_indices, val_indices, cal_indices = load_or_create_splits(raw_train)
    mean, std = calculate_mean_std(raw_train, train_indices)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 두 객체는 같은 원본 파일을 읽지만, 이미지를 꺼낼 때 서로 다른 변환을 적용한다.
    train_base = CIFAR10(root=DATA_DIR, train=True, download=False,
                         transform=train_transform)
    eval_base = CIFAR10(root=DATA_DIR, train=True, download=False,
                        transform=eval_transform)

    train_data = Subset(train_base, train_indices)
    val_data = Subset(eval_base, val_indices)
    cal_data = Subset(eval_base, cal_indices)
    test_data = CIFAR10(root=DATA_DIR, train=False, download=False,
                        transform=eval_transform)

    return train_data, val_data, cal_data, test_data


def make_data_loader():
    train_data, val_data, cal_data, test_data = make_datasets()

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=0)
    cal_loader = DataLoader(cal_data, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    return train_loader, val_loader, cal_loader, test_loader

def main():
    # train_data, val_data, cal_data, test_data = make_datasets()
    # print("train size:", len(train_data))
    # print("validation size:", len(val_data))
    # print("calibration size:", len(cal_data))
    # print("test size:", len(test_data))
    # image, label = train_data[0]
    # print("sample image shape:", tuple(image.shape), "label:", label)

    train_loader, val_loader, cal_loader, test_loader = make_data_loader()

    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)


if __name__ == "__main__":
    main()
