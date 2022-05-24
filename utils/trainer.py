import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from src.loss import NASLoss
from src.model import BaseModel, _vgg

# TODO: Implement the batch wise updation of sampling likelihoods. 
# Change the print and progress bar structure. 

class NASTrainer():

    def __init__(self, train_dataloader, validation_dataloader, test_dataloader, search_space, model_name, 
                num_classes = 100, init_weights = True, dropout = 0.5, batch_norm = True, 
                weights = None, progress = True, num_epochs = 180, learning_rate = 1e-4, weight_decay = 1e-4, 
                device = "cpu", optimizer_type = "Adam", criterion_type = "cross-entropy", temperature = 0.7,
                prob_dist = "maximum", eval_all = False, batch_update = True):

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

        self.criterion = NASLoss(criterion_type = self.criterion_type, temperature = self.temperature)

        self.num_configs = len(self.search_space)

        if self.prob_dist == "uniform":
            self.sample_probabilities = np.ones((self.num_configs), dtype = np.float32) / self.num_configs
        elif self.prob_dist == "maximum":
            self.sample_probabilities = np.ones((self.num_configs), dtype = np.float32)

    # NOTE: Everytime a new model is sampled, re initialise the optimizer    
    def set_optimizer(self, model):
        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)

        return optimizer
    
    def sample_architecture(self):

        sample_idx = np.argmax(self.sample_probabilities)
        all_idx = np.where(self.sample_probabilities == self.sample_probabilities[sample_idx])
        if all_idx[0].shape[0] == 1:
            return sample_idx, self.search_space[str(sample_idx)]
        else:
            random_idx = np.random.randint(0, all_idx[0].shape[0] - 1)
            sample_idx = all_idx[0][random_idx]
            return sample_idx, self.search_space[str(sample_idx)]

    def create_model(self, model, model_config):

        # TODO: Only implemented for VGG architecture, make it generalised
        # new_model = _vgg(self.search_space[str(sample_idx)], self.weights, batch_norm = self.batch_norm, progress = self.progress)
        # new_model.load_state_dict(model.state_dict)

        new_model = BaseModel(self.model_name, model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                        dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress)
        new_model = self.copy_parameters(model, new_model)
    
        return new_model

    def copy_parameters(self, model, new_model):

        for (name, param), (name_new, param_new) in zip(model.named_parameters(), new_model.named_parameters()):
            param_new.data = param.data

        return new_model

    def update_sampling_likelihood(self, model, sample_index, dataloader_type = "train", eval_all = False):
        # sample_probabilities[sample_idx] = sample
        # TODO: How to update the probabilities using the accuracy
        # NOTE: As of now, calculating accuracy of all the configurations
        # and updating the likelihood

        if eval_all:
            accuracies = []
            for idx, model_config in self.search_space.items():

                new_model = BaseModel(self.model_name, model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                                    dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress)
                new_model = self.copy_parameters(model, new_model)

                accuracy = self.evaluate(new_model, model_idx = idx)
                accuracies.append(accuracy)

            accuracies = np.array(accuracies)
            self.sample_probabilities = np.exp(self.sample_probabilities * accuracies)
            self.sample_probabilities = self.sample_probabilities / np.sum(self.sample_probabilities)
        else:
            self.sample_probabilities[sample_index] = self.evaluate(model, model_idx = sample_index)
            # NOTE: Do not normalise now, normalise it into a distribution after some number of epochs of training.
            # self.sample_probabilities = self.sample_probabilities / np.sum(self.sample_probabilities)

    def train(self):
        
        # TODO: Don't hardcode initial model configuration
        initial_model_config = self.search_space["0"]
        model = BaseModel(self.model_name, initial_model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                        dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress)
        model.to(self.device)

        for epoch in range(self.num_epochs):
            
            if epoch != 0 and not self.batch_update:
                sample_idx, model_config = self.sample_architecture()
                model = self.create_model(model, model_config)
                model.to(self.device)
            else:
                sample_idx = 0
                model_config = self.search_space["0"]
            
            optimizer = self.set_optimizer(model)
            total_loss = 0

            print(f"Training Model {sample_idx}")

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

                    if self.batch_update:
                        self.update_sampling_likelihood(model, sample_idx, dataloader_type = "train", eval_all = self.eval_all)
                        print("\n")
                        train_accuracy = self.evaluate(model, dataloader_type = "train", model_idx = str(sample_idx))
                        validation_accuracy = self.evaluate(model, dataloader_type = "val", model_idx = str(sample_idx))
                        test_accuracy = self.evaluate(model, dataloader_type = "test", model_idx = str(sample_idx))

                        print(f"\nModel {sample_idx}: \nTrain accuracy: {train_accuracy: .4f}, Validation accuracy: {validation_accuracy:.4f}, Test_accuracy: {test_accuracy: .4f}\n")

                        sample_idx, model_config = self.sample_architecture()
                        model = self.create_model(model, model_config)
                        model.to(self.device)

            # NOTE: Probability can be updated after every fixed number of batches instead of epochs
            if not self.batch_update:
                print("\n")
                self.update_sampling_likelihood(model, sample_idx, dataloader_type = "train", eval_all = self.eval_all)
                print("\n")
                train_accuracy = self.evaluate(model, dataloader_type = "train", model_idx = str(sample_idx))
                validation_accuracy = self.evaluate(model, dataloader_type = "val", model_idx = str(sample_idx))
                test_accuracy = self.evaluate(model, dataloader_type = "test", model_idx = str(sample_idx))

                print(f"\nModel {sample_idx}: \nTrain accuracy: {train_accuracy: .4f}, Validation accuracy: {validation_accuracy:.4f}, Test_accuracy: {test_accuracy: .4f}\n")
            

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
        return accuracy

    def get_best_configuration(self):

        best_config_index = np.argmax(self.sample_probabilities)
        best_config_index = np.where(self.sample_probabilities == self.sample_probabilities[best_config_index])
        
        best_configs = {}
        for idx in best_config_index[0]:
            best_configs[str(idx)] = self.search_space[str(idx)]
        
        return best_configs