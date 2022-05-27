import os
import sys
import torch
import argparse
import torchvision
import numpy as np
from src.model import cfgs
from src.utils import Logger
from utils.env import set_seed
from torchvision import transforms
from data.dataloader import CIFAR100
from torch.utils.data import DataLoader
from utils.search_space import PoolSearchSpace

from utils.trainer import Trainer, NASTrainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture_search", default = False, type = bool)
    parser.add_argument("--batch_norm", default = True, type = bool)
    parser.add_argument("--batch_sampling_size", default = 50, type = int)
    parser.add_argument("--batch_size", default = 64, type = int)
    parser.add_argument("--batch_update", default = False, type = bool)
    parser.add_argument("--criterion_type", default = "cross-entropy", type = str)
    parser.add_argument("--data_path", default = "./datasets/cifar100/", type = str)
    parser.add_argument("--dir_name", default = "vgg19", type = str)
    parser.add_argument("--discount_factor", default = 0.9, type = float)
    parser.add_argument("--download_data", default = False, type = bool)
    parser.add_argument("--dropout", default = 0.5, type = float)
    parser.add_argument("--exponential_moving_average", default = False, type = bool)
    parser.add_argument("--eval_all", default = False, type = bool)
    parser.add_argument("--init_weights", default = True, type = bool)
    parser.add_argument("--learning_rate", default = 1e-4, type = float)
    parser.add_argument("--log_dir", default = "./outputs/epoch_sample", type = str)
    parser.add_argument("--model_architecture", nargs = "+")
    parser.add_argument("--model_config", default = "E", type = str, choices = ["D", "E"])
    parser.add_argument("--model_name", default = "vgg19", type = str, choices = ["vgg16", "vgg19"])
    parser.add_argument("--num_classes", default = 100, type = int)
    parser.add_argument("--num_configs", default = 100, type = int)
    parser.add_argument("--num_epochs", default = 180, type = int)
    parser.add_argument("--num_repeats", default = 3, type = int)
    parser.add_argument("--num_val_examples", default = 1000, type = int)
    parser.add_argument("--num_workers", default = 2, type = int)
    parser.add_argument("--optimizer_type", default = "Adam", type = str)
    parser.add_argument("--progress", default = True, type = bool)
    parser.add_argument("--prob_dist", default = "maximum", type = str)
    parser.add_argument("--seed", default = 0, type = int)
    parser.add_argument("--temperature", default = 0.7, type = float)
    parser.add_argument("--visualisation_dir", default = "./visualisation/epoch_sample", type = str)
    parser.add_argument("--weight_decay", default = 1e-4, type = float)
    
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_out = os.path.join(args.log_dir, 'output.log')
    sys.stdout = Logger(log_out)
    
    print("\nArguments List:\n")
    for key, val in vars(args).items():
        print(f"{key}: {val}")
    print()

    set_seed(args.seed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

    train_data = CIFAR100(root = args.data_path, train = True, download = args.download_data, transform = transform)
    test_data = CIFAR100(root = args.data_path, train = False, download = args.download_data, transform = transform)
    
    train_data_len = len(train_data)
    random_permute = np.random.permutation(train_data_len)
    validation_data = [train_data[x] for x in random_permute[:args.num_val_examples]]
    train_data = [train_data[x] for x in random_permute[args.num_val_examples:]]
    
    train_dataloader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    validation_dataloader = DataLoader(validation_data, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    test_dataloader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

    if args.architecture_search:

        model_config = cfgs[args.model_config]
        pool_search_space = PoolSearchSpace("vgg19", model_config, num_configs = args.num_configs)
        pool_search_space.create_search_space()

        trainer = NASTrainer(train_dataloader, validation_dataloader, test_dataloader, pool_search_space.search_space, 
                    args.model_name, num_classes = args.num_classes, init_weights = args.init_weights, dropout = args.dropout, 
                    batch_norm = args.batch_norm, weights = None, progress = args.progress, num_epochs = args.num_epochs, 
                    learning_rate = args.learning_rate, weight_decay = args.weight_decay, device = device, 
                    optimizer_type = args.optimizer_type, criterion_type = args.criterion_type, temperature = args.temperature,
                    prob_dist = args.prob_dist, eval_all = args.eval_all, batch_update = args.batch_update, 
                    batch_sampling_size = args.batch_sampling_size, visualisation_dir = args.visualisation_dir, seed = args.seed,
                    exponential_moving_average = args.exponential_moving_average, discount_factor = args.discount_factor)
        trainer.train()

        best_pooling_configurations = trainer.get_best_configuration()
        for idx, architecture in best_pooling_configurations.items():
            print(f"{idx}: {architecture}")
        
        print("\n")

    else:

        for i in range(len(args.model_architecture)):
            if args.model_architecture[i].isdigit():
                args.model_architecture[i] = int(args.model_architecture[i])

        trainer = Trainer(train_dataloader, validation_dataloader, test_dataloader, args.model_architecture, args.model_name,
                    num_classes = args.num_classes, init_weights = args.init_weights, dropout = args.dropout, 
                    batch_norm = args.batch_norm, weights = None, progress = args.progress, num_epochs = args.num_epochs, 
                    learning_rate = args.learning_rate, weight_decay = args.weight_decay, device = device, 
                    optimizer_type = args.optimizer_type, criterion_type = args.criterion_type, temperature = args.temperature,
                    visualisation_dir = args.visualisation_dir, dir_name = args.dir_name)
        trainer.train()