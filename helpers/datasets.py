import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from PIL import Image
import random


class ImageListDataset(torch.utils.data.Dataset):
    """Dataset loaded from a text file with image paths and labels."""

    def __init__(self, list_file, root="", transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        with open(list_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                if root:
                    path = os.path.join(root, path)
                self.images.append(path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join("/data/ali/", self.images[idx])
        img = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label




def load_data(dataset):
    data_dir = '/dataset'
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_dir, train=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor()]))
        test_dataset = datasets.MNIST(data_dir, train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(data_dir, train=True,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()]))
        test_dataset = datasets.FashionMNIST(data_dir, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ]))
    elif dataset == "svhn":
        train_dataset = datasets.SVHN(data_dir, split="train",
                                      transform=transforms.Compose(
                                          [transforms.ToTensor()]))
        test_dataset = datasets.SVHN(data_dir, split="test",
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]))
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(data_dir, train=True,download=True,
                                         transform=transforms.Compose(
                                             [
                                                 transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                             ]))
        test_dataset = datasets.CIFAR10(data_dir, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]))
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(data_dir, train=True,
                                          transform=transforms.Compose(
                                              [
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                              ]))
        test_dataset = datasets.CIFAR100(data_dir, train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]))

    elif dataset == "tiny":
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ])
        }
        data_dir = "data/tiny-imagenet-200/"
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val', 'test']}
        train_dataset = image_datasets['train']
        test_dataset = image_datasets['val']
    else:
        raise NotImplementedError
    if dataset == "svhn":
        X_train, y_train = train_dataset.data, train_dataset.labels
        X_test, y_test = test_dataset.data, test_dataset.labels
    else:
        X_train, y_train = train_dataset.data, train_dataset.targets
        X_test, y_test = test_dataset.data, test_dataset.targets
    if "cifar10" in dataset or dataset == "svhn":
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        X_train = X_train.data.numpy()
        y_train = y_train.data.numpy()
        X_test = X_test.data.numpy()
        y_test = y_test.data.numpy()

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset




def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, partition, beta=0.4, num_users=5):
    n_parties = num_users
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(dataset)
    data_size = y_train.shape[0]

    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = np.unique(y_test).shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                # random [0.5963643 , 0.03712018, 0.04907753, 0.1115522 , 0.2058858 ]
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(   # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts


def partition_data_by_domain(source_files, target_file=None, transform=None):
    """Create datasets for domain adaptation where each domain is a client.

    Args:
        source_files (list[str]): list of txt files for source domains.
        target_file (str, optional): txt file for target domain used as test.
        transform: torchvision transform applied to all images.

    Returns:
        train_dataset (ConcatDataset): concatenation of all source domains.
        test_dataset (Dataset): dataset for target domain.
        net_dataidx_map (dict): indices for each client.
        train_data_cls_counts (dict): class distribution for each client.
    """
    if transform is None:
        transform = transforms.Compose([transforms.Resize(64),
                                       transforms.CenterCrop(64),
                                       transforms.ToTensor()])

    train_datasets = []
    net_dataidx_map = {}
    train_data_cls_counts = {}
    start = 0
    for cid, f in enumerate(source_files):
        ds = ImageListDataset(f, transform=transform)
        train_datasets.append(ds)
        idxs = list(range(start, start + len(ds)))
        net_dataidx_map[cid] = idxs
        labels = np.array(ds.labels)
        uniq, cnt = np.unique(labels, return_counts=True)
        train_data_cls_counts[cid] = {int(u): int(c) for u, c in zip(uniq, cnt)}
        start += len(ds)

    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    test_dataset = None
    if target_file is not None:
        test_dataset = ImageListDataset(target_file, transform=transform)

    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts
