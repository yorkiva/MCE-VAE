import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader

import argparse


def main(seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    
    so2_transform = transforms.Compose([
        transforms.RandomAffine(90, translate=(0., 0.)),
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))]  # mean and standard deviation, respectively, for normalization
    ])
    
    se2_transform = transforms.Compose([
        transforms.RandomAffine(90, translate=(0.25, 0.25)),
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))]  # mean and standard deviation, respectively, for normalization
    ])
    vanilla_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))])
    ])
    export_dataset(se2_transform, vanilla_transform, train=True, transform = 'se2')
    export_dataset(se2_transform, vanilla_transform, train=False, transform = 'se2')
    export_dataset(so2_transform, vanilla_transform, train=True, transform = 'so2')
    export_dataset(so2_transform, vanilla_transform, train=False, transform = 'so2')


def export_dataset(se2_transform, vanilla_transform, train: bool = True, transform = 'se2'):
    x, y = load_dataset(se2_transform, train=train)
    x_init, _ = load_dataset(vanilla_transform, train=train)
    name = transform + ('_train' if train else '_test')
    np.save(f'../mnist_{name}.npy', x)
    np.save(f'../mnist_init_{name}.npy', x_init)
    np.save(f'../mnist_target_{name}.npy', y)


def load_dataset(transform, train: bool = True) -> np.ndarray:
    dataset = datasets.MNIST('mnist', train=train, transform=transform, download=True)
    loader = DataLoader(dataset, shuffle=False, batch_size=len(dataset))

    dataset_torch = next(iter(loader))  # since batch_size = len(dataset), this returns the entire dataset
    data_array = dataset_torch[0].numpy()
    target_array = dataset_torch[1].numpy()
    return data_array, target_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', type=int, nargs=1, default=None, help='seed for the dataset generation')
    args = parser.parse_args()
    main(seed=args.seed)
