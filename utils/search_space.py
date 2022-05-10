import random
from copy import deepcopy

class PoolSearchSpace():

    def __init__(self, model_name, model_config, num_configs = 100):

        self.model_name = model_name
        self.model_config = model_config
        self.num_configs = num_configs
        self.search_space = {}

    def count_maxpool_layers(self):
        count = 0
        for layer in self.model_config:
            if layer == "M":
                count += 1
        return count

    def update_config(self, model_config, indices):

        for i in range(len(indices)):
            model_config.insert(indices[i] + i, "M")

        return model_config

    def create_search_space(self):
        
        num_maxpool_layers = self.count_maxpool_layers()

        layer_to_be_removed = "M"
        base_model_config = deepcopy(self.model_config)
        try:
            while True:
                base_model_config.remove(layer_to_be_removed)
        except ValueError:
            pass
        
        self.search_space["0"] = self.model_config
        model_config = deepcopy(base_model_config)

        for i in range(1, self.num_configs + 1):
        
            indices = random.sample(range(len(base_model_config)), num_maxpool_layers)
            model_config = self.update_config(model_config, sorted(indices))
            self.search_space[str(i)] = model_config
            model_config = deepcopy(base_model_config)