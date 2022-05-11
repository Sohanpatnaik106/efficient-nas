import os
import sys
import torch
import argparse
from src.model import cfgs
from utils.search_space import PoolSearchSpace
from utils.env import set_seed
from src.model import _vgg
from data.dataloader import CIFAR100
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":

    sys.path.append("..")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 128, type = int)
    parser.add_argument("--data_path", default = "./datasets/cifar100/", type = str)
    parser.add_argument("--download_data", default = False, type = bool)
    parser.add_argument("--model_config", default = "E", type = str, choices = ["D", "E"])
    parser.add_argument("--model_name", default = "vgg19", type = str, choices = ["vgg16", "vgg19"])
    parser.add_argument("--num_configs", default = 100, type = int)
    parser.add_argument("--num_workers", default = 2, type = int)
    parser.add_argument("--seed", default = 0, type = int)
    
    args = parser.parse_args()

    set_seed(args.seed)

    model_config = cfgs["E"]
    pool_search_space = PoolSearchSpace("vgg19", model_config, num_configs = 100)
    pool_search_space.create_search_space()
    

    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
    train_data = CIFAR100(root = args.data_path, train = True, download = args.download_data, transform = transform)
    test_data = CIFAR100(root = args.data_path, train = False, download = args.download_data, transform = transform)
    
    train_dataloader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    test_dataloader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

    
    model = _vgg(pool_search_space.search_space["50"], True, None, True)
    print(model)
