import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from src.loss import NASLoss
from src.model import BaseModel, _vgg

class NASTrainer():

    def __init__(self, train_dataloader, validation_dataloader, test_dataloader, model_name, search_space, 
                num_epochs = 180, learning_rate = 1e-4, weight_decay = 1e-4, device = "cpu", optimizer_type = "Adam"):

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.model_name = model_name
        self.search_space = search_space
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.optimizer_type = optimizer_type

        self.criterion = NASLoss()

        self.num_configs = len(self.search_space)
        self.sample_probabilities = np.ones((self.num_configs), dtype = np.float32) / self.num_configs

    # NOTE: Everytime a new model is sampled, re initialise the optimizer    
    def set_optimizer(self, model):
        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(lr = self.learning_rate, weight_decay = self.weight_decay)

        return optimizer
    
    def sample_architecture(self):

        sample_idx = np.argmax(self.sample_probabilities)
        return self.search_space[str(sample_idx)]

    def create_model(self, model, sample_idx):
        new_model = _vgg(self.search_space[str(idx)], batch_norm = True, progress = True)
        # TODO: Transfer the values of the weights from older model to newer model

        return new_model

    def update_sampling_likelihood(self, accuracy, sample_idx):
        # sample_probabilities[sample_idx] = sample
        # TODO: How to update the probabilities using the accuracy
        # NOTE: As of now, calculating accuracy of all the configurations
        # and updating the likelihood

    def train(self):
        
        model = BaseModel()
        model.to(self.device)

        for epoch in range(self.num_epochs):
            
            sample_idx = np.argmax(self.sample_probabilities)
            if epoch != 0:
                model = self.create_model(model, sample_idx)
                model.to(self.device)

            total_loss = 0
            with tqdm(train_dataloader, unit = "batch", position = 0, leave = True) as tepoch:
                for i, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

                    outputs = model(batch["images"])
                    loss = criterion(outputs)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    total_loss += loss.item()
    
            accuracy = self.evaluate(model)
            self.update_sampling_likelihood(accuracy, sample_idx)

        pass

    def evaluate(self):

        pass


