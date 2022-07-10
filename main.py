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
from data.dataloader import CIFAR100, CIFAR10
from torch.utils.data import DataLoader
from utils.search_space import PoolSearchSpace, HierarchicalSearchSpace

from utils.trainer import HNASTrainer, Trainer, NASTrainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture_search", default = False, type = bool)
    parser.add_argument("--batch_evaluate", default = False, type = bool)
    parser.add_argument("--batch_norm", default = False, type = bool)
    parser.add_argument("--batch_sampling_size", default = 50, type = int)
    parser.add_argument("--batch_size", default = 64, type = int)
    parser.add_argument("--batch_update", default = False, type = bool)
    parser.add_argument("--criterion_type", default = "cross-entropy", type = str)
    parser.add_argument("--data_path", default = "./datasets/cifar100/", type = str)
    parser.add_argument("--dataset", default = "cifar100", type = str)
    parser.add_argument("--dir_name", default = "vgg19", type = str)
    parser.add_argument("--discount_factor", default = 0.9, type = float)
    parser.add_argument("--distance_type", default = "euclidean", type = str)
    parser.add_argument("--download_data", default = False, type = bool)
    parser.add_argument("--dropout", default = 0.5, type = float)
    parser.add_argument("--dynamic_temperature", default = False, type = bool)
    parser.add_argument("--exponential_moving_average", default = False, type = bool)
    parser.add_argument("--eval_all", default = False, type = bool)
    parser.add_argument("--gpuid", default = 0, type = int)
    parser.add_argument("--hierarchical_search", default = False, type = bool)
    parser.add_argument("--init_weights", default = True, type = bool)
    parser.add_argument("--learning_rate", default = 1e-4, type = float)
    parser.add_argument("--learning_rate_scheduler", default = True, type = bool)
    parser.add_argument("--linkage_type", default = "single", type = str)
    parser.add_argument("--log_dir", default = "./outputs/hierarchical", type = str)
    parser.add_argument("--model_architecture", nargs = "+")
    parser.add_argument("--model_config", default = "E", type = str, choices = ["D", "E"])
    parser.add_argument("--model_name", default = "vgg19", type = str, choices = ["vgg16", "vgg19"])
    parser.add_argument("--normalise_prob_dist", default = False, type = bool)
    parser.add_argument("--num_classes", default = 100, type = int)
    parser.add_argument("--num_clusters", default = 5, type = int)
    parser.add_argument("--num_configs", default = 100, type = int)
    parser.add_argument("--num_epochs", default = 180, type = int)
    parser.add_argument("--num_repeats", default = 3, type = int)
    parser.add_argument("--num_val_examples", default = 1000, type = int)
    parser.add_argument("--num_workers", default = 2, type = int)
    parser.add_argument("--optimizer_type", default = "Adam", type = str)
    parser.add_argument("--progress", default = True, type = bool)
    parser.add_argument("--prob_dist", default = "maximum", type = str)
    parser.add_argument("--sample_binomial", default = True, type = bool)
    parser.add_argument("--seed", default = 0, type = int)
    parser.add_argument("--temperature", default = 10, type = float)
    parser.add_argument("--temperature_epoch_scaling", default = 50, type = float)
    parser.add_argument("--transform", default = True, type = bool)
    parser.add_argument("--track_running_stats", default = False, type = bool)
    parser.add_argument("--visualisation_dir", default = "./visualisation/hierarchical", type = str)
    parser.add_argument("--weight_decay", default = 1e-4, type = float)
    
    args = parser.parse_args()
    set_seed(args.seed)

    if not args.architecture_search:
        args.log_dir = os.path.join(args.log_dir, args.dir_name)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    log_out = os.path.join(args.log_dir, 'output.log')
    sys.stdout = Logger(log_out)
    
    print("\nArguments List:\n")
    for key, val in vars(args).items():
        print(f"{key}: {val}")
    print()

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else 'cpu')

    if args.transform:
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose([
                                transforms.RandomCrop(32, padding = 4, padding_mode = 'reflect'), 
                                transforms.RandomHorizontalFlip(), 
                                transforms.ToTensor(), 
                                transforms.Normalize(*stats, inplace = True)
                                ])
        test_transform = transforms.Compose([
                                transforms.ToTensor(), 
                                transforms.Normalize(*stats)
                                ])
    else:
        train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
        test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
    
    if args.dataset == "cifar100":
        train_data = CIFAR100(root = args.data_path, train = True, download = args.download_data, transform = train_transform)
        test_data = CIFAR100(root = args.data_path, train = False, download = args.download_data, transform = test_transform)
    
    elif args.dataset == "cifar10":
        train_data = CIFAR10(root = args.data_path, train = True, download = args.download_data, transform = train_transform)
        train_data = CIFAR10(root = args.data_path, train = False, download = args.download_data, transform = test_transform)
        
    train_data_len = len(train_data)
    random_permute = np.random.permutation(train_data_len)
    validation_data = [train_data[x] for x in random_permute[:args.num_val_examples]]
    train_data = [train_data[x] for x in random_permute[args.num_val_examples:]]
    
    train_dataloader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    validation_dataloader = DataLoader(validation_data, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    test_dataloader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

    train_accuracies, validation_accuracies, test_accuracies = [], [], []

    for r in range(args.num_repeats):

        print("************************************")
        print(f"* STARTING TRIAL {r+1}")
        print("************************************")

        args.seed += r
        set_seed(args.seed)

        if args.architecture_search:

            model_config = cfgs[args.model_config]
            pool_search_space = PoolSearchSpace(args.model_name, model_config, num_configs = args.num_configs)
            pool_search_space.create_search_space()

            trainer = NASTrainer(train_dataloader, validation_dataloader, test_dataloader, pool_search_space.search_space, 
                        args.model_name, num_classes = args.num_classes, init_weights = args.init_weights, dropout = args.dropout, 
                        batch_norm = args.batch_norm, weights = None, progress = args.progress, num_epochs = args.num_epochs, 
                        learning_rate = args.learning_rate, weight_decay = args.weight_decay, device = device, 
                        optimizer_type = args.optimizer_type, criterion_type = args.criterion_type, temperature = args.temperature,
                        prob_dist = args.prob_dist, eval_all = args.eval_all, batch_update = args.batch_update, 
                        batch_sampling_size = args.batch_sampling_size, visualisation_dir = args.visualisation_dir, seed = args.seed,
                        exponential_moving_average = args.exponential_moving_average, discount_factor = args.discount_factor,
                        normalise_prob_dist = args.normalise_prob_dist, track_running_stats = args.track_running_stats,
                        temperature_epoch_scaling = args.temperature_epoch_scaling, dynamic_temperature = args.dynamic_temperature, 
                        sample_binomial = args.sample_binomial, batch_evaluate = args.batch_evaluate, 
                        learning_rate_scheduler = args.learning_rate_scheduler)
            train_accuracy, validation_accuracy, test_accuracy = trainer.train()
            
            train_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)
            test_accuracies.append(test_accuracy)

            best_pooling_configurations = trainer.get_best_configuration()
            print("\nBest Pooling Configuration:\n")
            for idx, architecture in best_pooling_configurations.items():
                print(f"{idx}: {architecture}")
            
            print("\n")

        elif args.hierarchical_search:
            
            model_config = cfgs[args.model_config]
            hierarchical_search_space = HierarchicalSearchSpace(args.model_name, model_config, num_classes = args.num_classes, 
                                    num_configs = args.num_configs, init_weights = args.init_weights, device = device, 
                                    dropout = args.dropout, batch_norm = args.batch_norm, weights = None, progress = args.progress, 
                                    track_running_stats = args.track_running_stats, dataloader = train_dataloader, 
                                    batch_size = args.batch_size, distance_type = args.distance_type, linkage_type = args.linkage_type, 
                                    num_clusters = args.num_clusters, visualisation_dir = args.visualisation_dir)
            
            hierarchical_search_space.create_search_space()
            hierarchical_search_space.cluster_search_space()

            trainer = HNASTrainer(train_dataloader, validation_dataloader, test_dataloader, hierarchical_search_space.search_space, 
                args.model_name, num_classes = args.num_classes, init_weights = args.init_weights, dropout = args.dropout, 
                batch_norm = args.batch_norm, weights = None, progress = args.progress, num_epochs = args.num_epochs, 
                learning_rate = args.learning_rate, weight_decay = args.weight_decay, device = device, optimizer_type = args.optimizer_type, 
                criterion_type = args.criterion_type, temperature = args.temperature, prob_dist = args.prob_dist, eval_all = args.eval_all, 
                batch_update = args.batch_update, batch_sampling_size = args.batch_sampling_size, visualisation_dir = args.visualisation_dir,
                seed = args.seed, exponential_moving_average = args.exponential_moving_average, discount_factor = args.discount_factor, 
                normalise_prob_dist = args.normalise_prob_dist, track_running_stats = args.track_running_stats, 
                temperature_epoch_scaling = args.temperature_epoch_scaling, dynamic_temperature = args.dynamic_temperature, 
                cluster_tree = hierarchical_search_space.cluster_tree, cluster_root = hierarchical_search_space.cluster_root, 
                cluster_nodelist = hierarchical_search_space.cluster_nodelist, sample_binomial = args.sample_binomial,
                learning_rate_scheduler = args.learning_rate_scheduler)

            train_accuracy, validation_accuracy, test_accuracy = trainer.train()
            
            train_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)
            test_accuracies.append(test_accuracy)

            best_pooling_configurations = trainer.get_best_configuration()
            print("\nBest Pooling Configuration:\n")
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
                        visualisation_dir = args.visualisation_dir, dir_name = args.dir_name, learning_rate_scheduler = args.learning_rate_scheduler)
            train_accuracy, validation_accuracy, test_accuracy = trainer.train()

            train_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)
            test_accuracies.append(test_accuracy)

            for i in range(len(args.model_architecture)):
                args.model_architecture[i] = str(args.model_architecture[i])

    train_accuracies = np.array(train_accuracies)
    validation_accuracies = np.array(validation_accuracies)
    test_accuracies = np.array(test_accuracies)

    print(f"Summary of {args.num_repeats} experiment repeats:")
    print(f"MEAN\tTrain Accuracy: {np.mean(train_accuracies): .4f}, Validation Accuracy: {np.mean(validation_accuracies): .4f}, Test Accuracy: {np.mean(test_accuracies): .4f}")
    print(f"STDEV\tTrain Accuracy: {np.std(train_accuracies): .4f}, Validation Accuracy: {np.std(validation_accuracies): .4f}, Test Accuracy: {np.std(test_accuracies): .4f}")

    print("************************************")
    print(f"* END OF EXECUTION")
    print("************************************")
    print("\n")