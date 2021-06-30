import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

from tqdm import tqdm


# Methods for dealing with imbalanced datasets:
# 1. Oversampling
# 2. Class weighting

def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    class_weights = []
    count_all_files = 0
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(len(files))
            count_all_files += len(files)

    class_weights = [x / count_all_files for x in class_weights]

    print(class_weights)
    sample_weights = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader


def main():
    loader = get_loader(root_dir="data", batch_size=16)

    agnes_skinner = 0
    bart_simpson = 0
    for epoch in tqdm(range(5)):
        for data, labels in loader:
            agnes_skinner += torch.sum(labels == 0)
            bart_simpson += torch.sum(labels == 1)

    print(agnes_skinner)
    print(bart_simpson)


if __name__ == "__main__":
    main()
