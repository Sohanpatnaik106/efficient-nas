import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from random import choices
from src.loss import NASLoss
from src.model import BaseModel, _vgg
from .visualise import plot_sampling_prob_dist, plot_loss, plot_accuracy, plot_hierarchical_sampling_prob_dist

# TODO: Implement the batch wise updation of sampling likelihoods. 
# Change the print and progress bar structure.

class Trainer():

    def __init__(self, train_dataloader, validation_dataloader, test_dataloader, model_config, model_name, 
                num_classes = 100, init_weights = True, dropout = 0.5, batch_norm = True, 
                weights = None, progress = True, num_epochs = 180, learning_rate = 1e-4, weight_decay = 1e-4, 
                device = "cpu", optimizer_type = "Adam", criterion_type = "cross-entropy", temperature = 0.7,
                visualisation_dir = "./visualisation/base_model", dir_name = "vgg19", learning_rate_scheduler = False):

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

        self.model_name = model_name
        self.model_config = model_config
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.weights = weights
        self.progress = progress
        
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.optimizer_type = optimizer_type
        self.criterion_type = criterion_type
        self.temperature = temperature
        
        self.learning_rate_scheduler = learning_rate_scheduler
        self.scheduler = None
        
        self.dir_name = dir_name
        self.visualisation_dir = visualisation_dir
        if not os.path.exists(self.visualisation_dir):
            os.makedirs(self.visualisation_dir)
        self.visualisation_dir = os.path.join(self.visualisation_dir, self.dir_name)
        if not os.path.exists(self.visualisation_dir):
            os.makedirs(self.visualisation_dir)

        self.criterion = NASLoss(criterion_type = self.criterion_type, temperature = self.temperature)

    def set_optimizer(self, model):
        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        elif self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)

        if self.learning_rate_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            
        return optimizer

    def train(self):

        # TODO: Don't hardcode initial model configuration
        model = BaseModel(self.model_name, self.model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                        dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress)
        model.to(self.device)

        training_accuracies, validation_accuracies, test_accuracies = [], [], []
        training_losses, validation_losses, test_losses = [], [], []

        for epoch in range(self.num_epochs):
            
            optimizer = self.set_optimizer(model)
            total_loss = 0

            print(f"\nTraining Model\n")

            with tqdm(self.train_dataloader, unit = "batch", position = 0, leave = True) as tepoch:
                for i, (images, labels) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch + 1}")
                    
                    outputs = model(images.to(self.device))
                    loss = self.criterion(outputs.to(self.device), labels.to(self.device))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()
                    tepoch.set_postfix(loss = total_loss / (i+1))
            
            if self.learning_rate_scheduler:
                self.scheduler.step()
                  
            train_accuracy, train_loss = self.evaluate(model, dataloader_type = "train")
            validation_accuracy, validation_loss = self.evaluate(model, dataloader_type = "val")
            test_accuracy, test_loss = self.evaluate(model, dataloader_type = "test")
            
            print(f"\nModel {self.model_config}: \nTrain accuracy: {train_accuracy: .4f}, Validation accuracy: {validation_accuracy:.4f}, Test_accuracy: {test_accuracy: .4f}\n")

            training_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)
            test_accuracies.append(test_accuracy)
            training_losses.append(train_loss)
            validation_losses.append(validation_loss)
            test_losses.append(test_loss)

        plot_loss(training_losses, validation_losses, test_losses, self.visualisation_dir, num_epochs = self.num_epochs)
        plot_accuracy(training_accuracies, validation_accuracies, test_accuracies, self.visualisation_dir, num_epochs = self.num_epochs)        

        return training_accuracies[-1], validation_accuracies[-1], test_accuracies[-1]

    def evaluate(self, model, dataloader_type = "train"):

        if dataloader_type == "train":
            dataloader = self.train_dataloader
        elif dataloader_type == "val":
            dataloader = self.validation_dataloader
        elif dataloader_type == "test":
            dataloader = self.test_dataloader

        total_loss = 0
        total = 0
        correct = 0
        model.to(self.device)

        with tqdm(dataloader, unit = "batch", position = 0, leave = True) as tepoch:
            for i, (images, labels) in enumerate(tepoch):
                tepoch.set_description(f"Eval on {dataloader_type}")

                outputs = model(images.to(self.device))
                loss = self.criterion(outputs.to(self.device), labels.to(self.device))
                total_loss += loss.item()
                
                tepoch.set_postfix(loss = total_loss / (i+1))

                _, predicted = torch.max(outputs, dim = 1)
                total += labels.size(0)

                correct += (predicted == labels.to(self.device)).sum().item()
                tepoch.set_postfix(acc = correct / total)

        accuracy = correct / total
        loss = total_loss / (i+1)
        return accuracy, loss




class NASTrainer():

    def __init__(self, train_dataloader, validation_dataloader, test_dataloader, search_space, model_name, 
                num_classes = 100, init_weights = True, dropout = 0.5, batch_norm = True, 
                weights = None, progress = True, num_epochs = 180, learning_rate = 1e-4, weight_decay = 1e-4, 
                device = "cpu", optimizer_type = "Adam", criterion_type = "cross-entropy", temperature = 0.7,
                prob_dist = "maximum", eval_all = False, batch_update = True, batch_sampling_size = 30,
                visualisation_dir = "./visualisation/epoch_sample", seed = 0, exponential_moving_average = False, 
                discount_factor = 0.9, normalise_prob_dist = True, track_running_stats = False, temperature_epoch_scaling = 50,
                dynamic_temperature = True, sample_binomial = True, batch_evaluate = False, learning_rate_scheduler = False):

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.search_space = search_space
        self.model_name = model_name

        self.num_classes = num_classes
        self.init_weights = init_weights
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.weights = weights
        self.progress = progress
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.optimizer_type = optimizer_type
        self.criterion_type = criterion_type
        self.temperature = temperature
        self.prob_dist = prob_dist
        self.eval_all = eval_all
        self.batch_update = batch_update
        self.batch_sampling_size = batch_sampling_size
        self.seed = seed
        self.exponential_moving_average = exponential_moving_average
        self.discount_factor = discount_factor
        self.normalise_prob_dist = normalise_prob_dist
        self.track_running_stats = track_running_stats
        self.temperature_epoch_scaling = temperature_epoch_scaling
        self.dynamic_temperature = dynamic_temperature
        self.sample_binomial = sample_binomial
        self.batch_evaluate = batch_evaluate
        self.learning_rate_scheduler = learning_rate_scheduler
        
        self.scheduler = None

        self.visualisation_dir = visualisation_dir
        if not os.path.exists(self.visualisation_dir):
            os.makedirs(self.visualisation_dir)
        self.visualisation_dir = os.path.join(self.visualisation_dir, f"seed_{seed}")
        if not os.path.exists(self.visualisation_dir):
            os.makedirs(self.visualisation_dir)

        self.criterion = NASLoss(criterion_type = self.criterion_type, temperature = self.temperature)

        self.num_configs = len(self.search_space)

        if self.prob_dist == "uniform":
            self.sample_probabilities = np.ones((self.num_configs), dtype = np.float32) / self.num_configs
        elif self.prob_dist == "maximum":
            self.sample_probabilities = np.ones((self.num_configs), dtype = np.float32)

        self.indices = np.zeros((self.num_configs))
        for i in range(self.num_configs):
            self.indices[i] = i

    # NOTE: Everytime a new model is sampled, re initialise the optimizer    
    def set_optimizer(self, model):
        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        elif self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)

        if self.learning_rate_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            
        return optimizer

    def update_temperature(self, epoch_number):
        if self.temperature <= 0.7:
            self.temperature = 0.7
        else:
            self.temperature = self.temperature * np.exp(- epoch_number / self.temperature_epoch_scaling)
    
    def sample_architecture(self):
        
        if self.sample_binomial:
            sample_probabilities = self.sample_probabilities
            if self.normalise_prob_dist:
                sample_probabilities = np.exp(sample_probabilities / self.temperature) \
                                            / np.sum(np.exp(sample_probabilities / self.temperature))
            
            sample_idx = choices(self.indices, sample_probabilities, k = 1)
            sample_idx = int(sample_idx[0])
            return sample_idx, self.search_space[str(sample_idx)]
        
        else: 

            sample_probabilities = self.sample_probabilities
            if self.normalise_prob_dist:
                sample_probabilities = np.exp(sample_probabilities / self.temperature) \
                                            / np.sum(np.exp(sample_probabilities / self.temperature))
            
            sample_idx = np.argmax(sample_probabilities)
            all_idx = np.where(sample_probabilities == sample_probabilities[sample_idx])
            if all_idx[0].shape[0] == 1:
                return sample_idx, self.search_space[str(sample_idx)]
            else:
                random_idx = np.random.randint(0, all_idx[0].shape[0])
                sample_idx = all_idx[0][random_idx]
                return sample_idx, self.search_space[str(sample_idx)]

    def create_model(self, model, model_config):

        # TODO: Only implemented for VGG architecture, make it generalised
        # new_model = _vgg(self.search_space[str(sample_idx)], self.weights, batch_norm = self.batch_norm, progress = self.progress)
        # new_model.load_state_dict(model.state_dict)

        new_model = BaseModel(self.model_name, model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                        dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress,
                        track_running_stats = self.track_running_stats)
        new_model = self.copy_parameters(model, new_model)

        del model
        return new_model

    def copy_parameters(self, model, new_model):

        for (name, param), (name_new, param_new) in zip(model.named_parameters(), new_model.named_parameters()):
            param_new.data = param.data

        del model
        return new_model

    def update_sampling_likelihood(self, model, sample_index, dataloader_type = "train", eval_all = False, eval_idx = 0):
        # sample_probabilities[sample_idx] = sample
        # TODO: How to update the probabilities using the accuracy
        # NOTE: As of now, calculating accuracy of all the configurations
        # and updating the likelihood
        
        # TODO: Update the eval all part 
        if eval_all:
            accuracies = []
            for idx, model_config in self.search_space.items():

                new_model = BaseModel(self.model_name, model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                                    dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress)
                new_model = self.copy_parameters(model, new_model)

                accuracy, loss = self.evaluate(new_model, model_idx = idx, eval_idx = eval_idx)
                accuracies.append(accuracy)

            accuracies = np.array(accuracies)
            self.sample_probabilities = np.exp(self.sample_probabilities * accuracies)
            self.sample_probabilities = self.sample_probabilities / np.sum(self.sample_probabilities)
        else:
            accuracy, loss = self.evaluate(model, model_idx = sample_index, eval_idx = eval_idx)
            if self.exponential_moving_average:
                self.sample_probabilities[sample_index] = self.discount_factor * self.sample_probabilities[sample_index] \
                                                                        + (1 - self.discount_factor) * accuracy
            elif not self.exponential_moving_average:
                self.sample_probabilities[sample_index] = accuracy

            # NOTE: Do not normalise now, normalise it into a distribution after some number of epochs of training.
            # if self.normalise_prob_dist:
            #     acc = np.ones(self.sample_probabilities.shape[0])
            #     if self.exponential_moving_average:
            #         acc[sample_index] = accuracy
            #     self.sample_probabilities = np.exp((self.sample_probabilities * acc) / self.temperature) \
            #                                             / np.sum(np.exp((self.sample_probabilities * acc) / self.temperature))

            """
                1, 1, 1, 1, 1
                1, 0.905, 1, 1, 1 ==> 0.2, 0.18, 0.2, 0.2, 0.2
            
            # NOTE: Sample with normalised probabilities but update using the previous values. 

            """

    def train(self):
        
        # TODO: Don't hardcode initial model configuration
        # initial_model_config = self.search_space["0"]
        sample_idx = choices(self.indices, self.sample_probabilities, k = 1)
        sample_idx = int(sample_idx[0])
        initial_model_config = self.search_space[str(sample_idx)]

        model = BaseModel(self.model_name, initial_model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                        dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress,
                        track_running_stats = self.track_running_stats)
        model.to(self.device)

        training_accuracies, validation_accuracies, test_accuracies = [], [], []
        training_losses, validation_losses, test_losses = [], [], []

        for epoch in range(self.num_epochs):
            
            if epoch != 0 and not self.batch_update:
                sample_idx, model_config = self.sample_architecture()
                model = self.create_model(model, model_config)
                model.to(self.device)
            else:
                model_config = self.search_space[str(sample_idx)]
            
            optimizer = self.set_optimizer(model)
            total_loss = 0

            print(f"\nTraining Model {sample_idx}")

            with tqdm(self.train_dataloader, unit = "batch", position = 0, leave = True) as tepoch:
                for i, (images, labels) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch + 1}")
                    
                    outputs = model(images.to(self.device))
                    loss = self.criterion(outputs.to(self.device), labels.to(self.device))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()
                    tepoch.set_postfix(loss = total_loss / (i+1))

                    if self.batch_update and (i+1) % self.batch_sampling_size == 0:
                        self.update_sampling_likelihood(model, sample_idx, dataloader_type = "train", eval_all = self.eval_all, eval_idx = i)
                        print("\n")
                        train_accuracy, train_loss = self.evaluate(model, dataloader_type = "train", model_idx = str(sample_idx), eval_idx = i)
                        validation_accuracy, validation_loss = self.evaluate(model, dataloader_type = "val", model_idx = str(sample_idx))
                        test_accuracy, test_loss = self.evaluate(model, dataloader_type = "test", model_idx = str(sample_idx))

                        print(f"\nModel {sample_idx}: \nTrain accuracy: {train_accuracy: .4f}, Validation accuracy: {validation_accuracy:.4f}, Test_accuracy: {test_accuracy: .4f}\n")

                        sample_idx, model_config = self.sample_architecture()
                        model = self.create_model(model, model_config)
                        model.to(self.device)

            if self.learning_rate_scheduler:
                self.scheduler.step()
                
            # NOTE: Probability can be updated after every fixed number of batches instead of epochs
            # if not self.batch_update:
            print("\n")
            self.update_sampling_likelihood(model, sample_idx, dataloader_type = "train", eval_all = self.eval_all)
            print("\n")
            train_accuracy, train_loss = self.evaluate(model, dataloader_type = "train", model_idx = str(sample_idx))
            validation_accuracy, validation_loss = self.evaluate(model, dataloader_type = "val", model_idx = str(sample_idx))
            test_accuracy, test_loss = self.evaluate(model, dataloader_type = "test", model_idx = str(sample_idx))

            print(f"\nModel {sample_idx}: \nTrain accuracy: {train_accuracy: .4f}, Validation accuracy: {validation_accuracy:.4f}, Test_accuracy: {test_accuracy: .4f}\n")

            training_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)
            test_accuracies.append(test_accuracy)
            training_losses.append(train_loss)
            validation_losses.append(validation_loss)
            test_losses.append(test_loss)

            # Plot the probability distribution after every epoch
            plot_sampling_prob_dist(self.sample_probabilities, epoch+1, self.num_configs, self.visualisation_dir, 
                                    normalise = self.normalise_prob_dist, temperature = self.temperature)
            if self.dynamic_temperature:
                self.update_temperature(epoch)

        plot_loss(training_losses, validation_losses, test_losses, self.visualisation_dir, num_epochs = self.num_epochs)
        plot_accuracy(training_accuracies, validation_accuracies, test_accuracies, self.visualisation_dir, num_epochs = self.num_epochs)

        best_configs = self.get_best_configuration()
        for idx, config in best_configs.items():
            model = self.create_model(model, config)
            train_accuracy, train_loss = self.evaluate(model, dataloader_type = "train", model_idx = idx)
            validation_accuracy, validation_loss = self.evaluate(model, dataloader_type = "val", model_idx = idx)
            test_accuracy, test_loss = self.evaluate(model, dataloader_type = "test", model_idx = idx)
            break

        return train_accuracy, validation_accuracy, test_accuracy
    
    def evaluate(self, model, dataloader_type = "train", model_idx = "0", eval_idx = -1):

        if dataloader_type == "train":
            dataloader = self.train_dataloader
        elif dataloader_type == "val":
            dataloader = self.validation_dataloader
        elif dataloader_type == "test":
            dataloader = self.test_dataloader

        total_loss = 0
        total = 0
        correct = 0
        model.to(self.device)

        with tqdm(dataloader, unit = "batch", position = 0, leave = True) as tepoch:
            for i, (images, labels) in enumerate(tepoch):
                tepoch.set_description(f"Model {model_idx}: Eval on {dataloader_type}")
                
                if self.batch_evaluate and eval_idx >= 0 and i != eval_idx:
                    # print("here")
                    continue

                # print(self.batch_evaluate)
                # print(eval_idx)
                # print(i)
                outputs = model(images.to(self.device))
                loss = self.criterion(outputs.to(self.device), labels.to(self.device))
                total_loss += loss.item()
                
                tepoch.set_postfix(loss = total_loss / (i+1))

                _, predicted = torch.max(outputs, dim = 1)
                total += labels.size(0)

                correct += (predicted == labels.to(self.device)).sum().item()
                tepoch.set_postfix(acc = correct / total)

        if not self.batch_evaluate:
            loss = total_loss / (i+1)
        
        accuracy = correct / total
        # loss = total_loss / (i+1)
        return accuracy, loss

    def get_best_configuration(self):

        best_config_index = np.argmax(self.sample_probabilities)
        best_config_index = np.where(self.sample_probabilities == self.sample_probabilities[best_config_index])
        
        best_configs = {}
        for idx in best_config_index[0]:
            best_configs[str(idx)] = self.search_space[str(idx)]
        
        return best_configs



















# TODO: Take as input, the hierarchy of the architectures, update the sampling function and likelihood update.

class HNASTrainer():

    def __init__(self, train_dataloader, validation_dataloader, test_dataloader, search_space, model_name, 
                num_classes = 100, init_weights = True, dropout = 0.5, batch_norm = True, 
                weights = None, progress = True, num_epochs = 180, learning_rate = 1e-4, weight_decay = 1e-4, 
                device = "cpu", optimizer_type = "Adam", criterion_type = "cross-entropy", temperature = 0.7,
                prob_dist = "maximum", eval_all = False, batch_update = True, batch_sampling_size = 30,
                visualisation_dir = "./visualisation/epoch_sample", seed = 0, exponential_moving_average = False, 
                discount_factor = 0.9, normalise_prob_dist = True, track_running_stats = False, temperature_epoch_scaling = 50,
                dynamic_temperature = True, cluster_tree = None, cluster_root = None, cluster_nodelist = None,
                sample_binomial = True, learning_rate_scheduler = False):

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.search_space = search_space
        self.model_name = model_name

        self.num_classes = num_classes
        self.init_weights = init_weights
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.weights = weights
        self.progress = progress
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.optimizer_type = optimizer_type
        self.criterion_type = criterion_type
        self.temperature = temperature
        self.prob_dist = prob_dist
        self.eval_all = eval_all
        self.batch_update = batch_update
        self.batch_sampling_size = batch_sampling_size
        self.seed = seed
        self.exponential_moving_average = exponential_moving_average
        self.discount_factor = discount_factor
        self.normalise_prob_dist = normalise_prob_dist
        self.track_running_stats = track_running_stats
        self.temperature_epoch_scaling = temperature_epoch_scaling
        self.dynamic_temperature = dynamic_temperature
        self.learning_rate_scheduler = learning_rate_scheduler
        
        self.scheduler = None

        self.cluster_tree = cluster_tree
        self.cluster_root = cluster_root
        self.cluster_nodelist = cluster_nodelist

        self.sample_binomial = sample_binomial

        self.visualisation_dir = visualisation_dir
        if not os.path.exists(self.visualisation_dir):
            os.makedirs(self.visualisation_dir)
        self.visualisation_dir = os.path.join(self.visualisation_dir, f"seed_{seed}")
        if not os.path.exists(self.visualisation_dir):
            os.makedirs(self.visualisation_dir)

        self.criterion = NASLoss(criterion_type = self.criterion_type, temperature = self.temperature)

        self.num_configs = len(self.search_space)

        if self.prob_dist == "uniform":
            self.sample_probabilities = np.ones((self.num_configs), dtype = np.float32) / self.num_configs
        elif self.prob_dist == "maximum":
            self.sample_probabilities = np.ones((self.num_configs), dtype = np.float32)

        self.sampled_nodes = []
        self.sample_idx = None

    # NOTE: Everytime a new model is sampled, re initialise the optimizer    
    def set_optimizer(self, model):
        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)

        elif self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)

        if self.learning_rate_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
        return optimizer

    def update_temperature(self, epoch_number):
        if self.temperature <= 0.7:
            self.temperature = 0.7
        else:
            self.temperature = self.temperature * np.exp(- epoch_number / self.temperature_epoch_scaling)
    
    # NOTE: Implement this function using the cluster tree, cluster node

    def sample_nodes(self, node):

        self.sampled_nodes.append(node)
        if node.get_id() <= self.num_configs:
            self.sample_idx = node.get_id()

        if not node.get_left() and not node.get_right():
            return 

        sample_probabilities = np.array([node.get_left().get_sample_probability(), node.get_right().get_sample_probability()])
        indices = np.array([0, 1])
        sample_idx = choices(indices, sample_probabilities, k = 1)

        if sample_idx[0] == 0:
            self.sample_nodes(node.get_left())
        else:
            self.sample_nodes(node.get_right())

    def sample_architecture(self):

        if self.sample_binomial:
            self.sampled_nodes = []
            self.sample_nodes(self.cluster_root)
            return self.sample_idx, self.search_space[str(self.sample_idx)]
        
        # TODO: Implement sampling with maximum probability, not likely
        else: 
            raise NotImplementedError

    def create_model(self, model, model_config):

        # TODO: Only implemented for VGG architecture, make it generalised
        # new_model = _vgg(self.search_space[str(sample_idx)], self.weights, batch_norm = self.batch_norm, progress = self.progress)
        # new_model.load_state_dict(model.state_dict)

        new_model = BaseModel(self.model_name, model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                        dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress,
                        track_running_stats = self.track_running_stats)
        new_model = self.copy_parameters(model, new_model)

        del model    
        return new_model

    def copy_parameters(self, model, new_model):

        for (name, param), (name_new, param_new) in zip(model.named_parameters(), new_model.named_parameters()):
            param_new.data = param.data

        del model
        return new_model

    # NOTE: Update this function using cluster tree and cluster root
    def update_sampling_likelihood(self, model, sample_index, dataloader_type = "train", eval_all = False):
        # sample_probabilities[sample_idx] = sample
        # TODO: How to update the probabilities using the accuracy
        # NOTE: As of now, calculating accuracy of all the configurations
        # and updating the likelihood
        
        # TODO: Implement the eval all part 
        if eval_all:
            raise NotImplementedError

        else:
            accuracy, loss = self.evaluate(model, model_idx = sample_index)
            for node in self.sampled_nodes:

                if self.exponential_moving_average:
                    node.sample_probability = self.discount_factor * node.sample_probability + (1 - self.discount_factor) * accuracy
                elif not self.exponential_moving_average:
                    node.sample_probability = accuracy

            # TODO: Implement normalisation of probabilities at each level
            
            # NOTE: Do not normalise now, normalise it into a distribution after some number of epochs of training.
            # if self.normalise_prob_dist:
            #     acc = np.ones(self.sample_probabilities.shape[0])
            #     if self.exponential_moving_average:
            #         acc[sample_index] = accuracy
            #     self.sample_probabilities = np.exp((self.sample_probabilities * acc) / self.temperature) \
            #                                             / np.sum(np.exp((self.sample_probabilities * acc) / self.temperature))

    def train(self):

        sample_idx, initial_model_config = self.sample_architecture()

        model = BaseModel(self.model_name, initial_model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                        dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress,
                        track_running_stats = self.track_running_stats)
        model.to(self.device)

        training_accuracies, validation_accuracies, test_accuracies = [], [], []
        training_losses, validation_losses, test_losses = [], [], []

        for epoch in range(self.num_epochs):
            
            if epoch != 0 and not self.batch_update:
                sample_idx, model_config = self.sample_architecture()
                model = self.create_model(model, model_config)
                model.to(self.device)
            else:
                model_config = self.search_space[str(sample_idx)]
            
            optimizer = self.set_optimizer(model)
            total_loss = 0

            print(f"\nTraining Model {sample_idx}")

            with tqdm(self.train_dataloader, unit = "batch", position = 0, leave = True) as tepoch:
                for i, (images, labels) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch + 1}")
                    
                    outputs = model(images.to(self.device))
                    loss = self.criterion(outputs.to(self.device), labels.to(self.device))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()
                    tepoch.set_postfix(loss = total_loss / (i+1))

                    if self.batch_update and (i+1) % self.batch_sampling_size == 0:
                        self.update_sampling_likelihood(model, sample_idx, dataloader_type = "train", eval_all = self.eval_all)
                        print("\n")
                        train_accuracy, train_loss = self.evaluate(model, dataloader_type = "train", model_idx = str(sample_idx))
                        validation_accuracy, validation_loss = self.evaluate(model, dataloader_type = "val", model_idx = str(sample_idx))
                        test_accuracy, test_loss = self.evaluate(model, dataloader_type = "test", model_idx = str(sample_idx))

                        print(f"\nModel {sample_idx}: \nTrain accuracy: {train_accuracy: .4f}, Validation accuracy: {validation_accuracy:.4f}, Test_accuracy: {test_accuracy: .4f}\n")

                        sample_idx, model_config = self.sample_architecture()
                        model = self.create_model(model, model_config)
                        model.to(self.device)

            if self.learning_rate_scheduler:
                self.scheduler.step()
            
            # NOTE: Probability can be updated after every fixed number of batches instead of epochs
            # if not self.batch_update:
            print("\n")
            self.update_sampling_likelihood(model, sample_idx, dataloader_type = "train", eval_all = self.eval_all)
            print("\n")
            train_accuracy, train_loss = self.evaluate(model, dataloader_type = "train", model_idx = str(sample_idx))
            validation_accuracy, validation_loss = self.evaluate(model, dataloader_type = "val", model_idx = str(sample_idx))
            test_accuracy, test_loss = self.evaluate(model, dataloader_type = "test", model_idx = str(sample_idx))

            print(f"\nModel {sample_idx}: \nTrain accuracy: {train_accuracy: .4f}, Validation accuracy: {validation_accuracy:.4f}, Test_accuracy: {test_accuracy: .4f}\n")

            training_accuracies.append(train_accuracy)
            validation_accuracies.append(validation_accuracy)
            test_accuracies.append(test_accuracy)
            training_losses.append(train_loss)
            validation_losses.append(validation_loss)
            test_losses.append(test_loss)

            # Plot the probability distribution after every epoch
            plot_hierarchical_sampling_prob_dist(self.cluster_nodelist, epoch+1, self.num_configs, self.visualisation_dir, normalise = self.normalise_prob_dist, temperature = self.temperature)
            if self.dynamic_temperature:
                self.update_temperature(epoch)

        plot_loss(training_losses, validation_losses, test_losses, self.visualisation_dir, num_epochs = self.num_epochs)
        plot_accuracy(training_accuracies, validation_accuracies, test_accuracies, self.visualisation_dir, num_epochs = self.num_epochs)

        best_configs = self.get_best_configuration()
        for idx, config in best_configs.items():
            model = self.create_model(model, config)
            train_accuracy, train_loss = self.evaluate(model, dataloader_type = "train", model_idx = idx)
            validation_accuracy, validation_loss = self.evaluate(model, dataloader_type = "val", model_idx = idx)
            test_accuracy, test_loss = self.evaluate(model, dataloader_type = "test", model_idx = idx)
            break

        return train_accuracy, validation_accuracy, test_accuracy
    
    
    def evaluate(self, model, dataloader_type = "train", model_idx = "0"):

        if dataloader_type == "train":
            dataloader = self.train_dataloader
        elif dataloader_type == "val":
            dataloader = self.validation_dataloader
        elif dataloader_type == "test":
            dataloader = self.test_dataloader

        total_loss = 0
        total = 0
        correct = 0
        model.to(self.device)

        with tqdm(dataloader, unit = "batch", position = 0, leave = True) as tepoch:
            for i, (images, labels) in enumerate(tepoch):
                tepoch.set_description(f"Model {model_idx}: Eval on {dataloader_type}")

                outputs = model(images.to(self.device))
                loss = self.criterion(outputs.to(self.device), labels.to(self.device))
                total_loss += loss.item()
                
                tepoch.set_postfix(loss = total_loss / (i+1))

                _, predicted = torch.max(outputs, dim = 1)
                total += labels.size(0)

                correct += (predicted == labels.to(self.device)).sum().item()
                tepoch.set_postfix(acc = correct / total)

        accuracy = correct / total
        loss = total_loss / (i+1)
        return accuracy, loss

    def get_best_configuration(self):

        best_config_index = np.argmax(self.sample_probabilities)
        best_config_index = np.where(self.sample_probabilities == self.sample_probabilities[best_config_index])
        
        best_configs = {}
        for idx in best_config_index[0]:
            best_configs[str(idx)] = self.search_space[str(idx)]
        
        return best_configs