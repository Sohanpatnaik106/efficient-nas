import random
from copy import deepcopy
from .clustering import HierarchicalClustering

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

# TODO: Update hirarchical clustering based on distance matrix required for the architectures. 


class HierarchicalSearchSpace():

    def __init__(self, model_name, model_config, num_classes = 100, num_configs = 1000, init_weights = True, device = "cpu",
				dropout = 0.5, batch_norm = True, weights = None, progress = True, track_running_stats = False, 
                dataloader = None, batch_size = 256, distance_type = "euclidean", linkage_type = "single", num_clusters = 5):

        self.model_name = model_name
        self.model_config = model_config
        self.num_classes = num_classes
        self.num_configs = num_configs
        self.init_weights = init_weights
        self.device = device
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.weights = weights
        self.progress = progress
        self.track_running_stats = track_running_stats
        self.dataloader = dataloader
        
        self.batch_size = batch_size
        self.distance_type = distance_type
        self.linkage_type = linkage_type

        self.num_clusters = num_clusters

        self.search_space = {}
        self.cluster_dict = {}
        self.distance_matrix = None


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

    def cluster_search_space(self):

        hierarchical_clustering = HierarchicalClustering(self.model_name, num_classes = self.num_classes, init_weights = self.init_weights, device = self.device,
					                    dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress, 
					                    track_running_stats = self.track_running_stats, dataloader = self.dataloader, search_space = self.search_space, 
                                        batch_size = self.batch_size, distance_type = self.distance_type, linkage_type = self.linkage_type, 
                                        num_clusters = self.num_clusters)

        self.cluster_dict, self.distance_matrix = hierarchical_clustering.cluster()
        print(self.cluster_dict)
        print(self.distance_matrix)