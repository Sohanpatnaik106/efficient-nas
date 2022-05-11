import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from src.loss import NASLoss
from src.model import BaseModel, _vgg

class NASTrainer():

    def __init__(self, train_dataloader, validation_dataloader, test_dataloader, search_space, model_name, 
                num_classes = 100, init_weights = True, dropout = 0.5, batch_norm = True, 
                weights = None, progress = True, num_epochs = 180, learning_rate = 1e-4, weight_decay = 1e-4, 
                device = "cpu", optimizer_type = "Adam", criterion_type = "cross-entropy", temperature = 0.7):

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

        self.criterion = NASLoss(criterion_type = self.criterion_type, temperature = self.temperature)

        self.num_configs = len(self.search_space)
        self.sample_probabilities = np.ones((self.num_configs), dtype = np.float32) / self.num_configs

    # NOTE: Everytime a new model is sampled, re initialise the optimizer    
    def set_optimizer(self, model):
        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)

        return optimizer
    
    def sample_architecture(self):

        sample_idx = np.argmax(self.sample_probabilities)
        return self.search_space[str(sample_idx)]

    def create_model(self, model, model_config):

        # TODO: Only implemented for VGG architecture, make it generalised
        # new_model = _vgg(self.search_space[str(sample_idx)], self.weights, batch_norm = self.batch_norm, progress = self.progress)
        # new_model.load_state_dict(model.state_dict)

        new_model = BaseModel(self.model_name, model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                        dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress)
        new_model.model.load_state_dict(model.model.load_state_dict)
    
        return new_model

    def update_sampling_likelihood(self, model, dataloader_type = "train"):
        # sample_probabilities[sample_idx] = sample
        # TODO: How to update the probabilities using the accuracy
        # NOTE: As of now, calculating accuracy of all the configurations
        # and updating the likelihood

        accuracies = []
        for model_config in self.search_space:
            new_model = BaseModel(self.model_name, model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                                dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress)
            new_model.model.load_state_dict(model.model.load_state_dict)
            accuracy = self.evaluate(new_model)
            accuracies.append(accuracy)

        accuracies = np.array(accuracies)
        self.sample_probabilities = np.exp(self.sample_probabilities * accuracies)
        self.sample_probabilities = self.sample_probabilities / np.sum(self.sample_probabilities)

    def train(self):
        
        # TODO: Don't hardcode initial model configuration
        initial_model_config = self.search_space["0"]
        model = BaseModel(self.model_name, initial_model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
                        dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress)
        model.to(self.device)

        for epoch in range(self.num_epochs):
            
            if epoch != 0:
                model_config = self.sample_architecture()
                model = self.create_model(model, model_config)
                model.to(self.device)
            else:
                model_config = self.search_space["0"]
            
            optimizer = self.set_optimizer(model)
            total_loss = 0

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
                
            self.update_sampling_likelihood(model, dataloader_type = "train")
            train_accuracy = self.evaluate(model, dataloader_type = "train")
            validation_accuracy = self.evaluate(model, dataloader_type = "val")
            test_accuracy = self.evaluate(model, dataloader_type = "test")

            print(f"Train accuracy: {train_accuracy}, validation accuracy: {validation_accuracy}, test_accuracy: {test_accuracy}")
            
    def evaluate(self, model, dataloader_type = "train"):

        if dataloader_type == "train":
            dataloader = self.train_dataloader
        elif dataloader_type == "val":
            dataloader = self.validation_dataloader
        elif dataloader_type == "test":
            dataloader = self.test_dataloader

        total = 0
        with tqdm(dataloader, unit = "batch", position = 0, leave = True) as tepoch:
            for i, (images, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch + 1}")
                
                outputs = model(images.to(self.device))
                loss = criterion(outputs.to(self.device), labels.to(self.device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                tepoch.set_postfix(loss = total_loss / (i+1))

                _, predicted = torch.max(outputs, dim = 1)
                total += labels.size(0)

                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy