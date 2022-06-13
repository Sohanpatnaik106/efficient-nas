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

# TODO: Update hirarchical clustering based on distance matrix required for the architectures. 


"""
Executing code: 
Python hclust.py iris.dat 3
"""

"""
Change log: 
- Nov 8, 2015
1. Change the logic to calculation centroid
2. Add judgement for some invalid input cases
"""

# import sys
# import math
# import os
# from src.utils import *
# import itertools

# class Hierarchical_Clustering:

#     def __init__(self, ipt_data, ipt_k):
    
#         self.input_file_name = ipt_data
#         self.k = ipt_k
#         self.dataset = None
#         self.dataset_size = 0
#         self.dimension = 0
#         self.heap = []
#         self.clusters = []
#         self.gold_standard = {}

#     def initialize(self):
#         """
#         Initialize and check parameters
#         """
#         # check file exist and if it's a file or dir
#         if not os.path.isfile(self.input_file_name):
#             self.quit("Input file doesn't exist or it's not a file")

#         self.dataset, self.clusters, self.gold_standard = self.load_data(self.input_file_name)
#         self.dataset_size = len(self.dataset)

#         if self.dataset_size == 0:
#             self.quit("Input file doesn't include any data")

#         if self.k == 0:
#             self.quit("k = 0, no cluster will be generated")

#         if self.k > self.dataset_size:
#             self.quit("k is larger than the number of existing clusters")

#         self.dimension = len(self.dataset[0]["data"])

#         if self.dimension == 0:
#             self.quit("dimension for dataset cannot be zero")

#     """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#     """                      Hierarchical Clustering Functions                       """
#     """                                                                              """    
#     """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#     def euclidean_distance(self, data_point_one, data_point_two):
#         """
#         euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
#         assume that two data points have same dimension
#         """
#         size = len(data_point_one)
#         result = 0.0
#         for i in range(size):
#             f1 = float(data_point_one[i])   # feature for data one
#             f2 = float(data_point_two[i])   # feature for data two
#             tmp = f1 - f2
#             result += pow(tmp, 2)
#         result = math.sqrt(result)
#         return result

#     def compute_pairwise_distance(self, dataset):
#         result = []
#         dataset_size = len(dataset)
#         for i in range(dataset_size-1):    # ignore last i
#             for j in range(i+1, dataset_size):     # ignore duplication
#                 dist = self.euclidean_distance(dataset[i]["data"], dataset[j]["data"])

#                 # duplicate dist, need to be remove, and there is no difference to use tuple only
#                 # leave second dist here is to take up a position for tie selection
#                 result.append( (dist, [dist, [[i], [j]]]) )

#         return result
                
#     def build_priority_queue(self, distance_list):
#         heapq.heapify(distance_list)
#         self.heap = distance_list
#         return self.heap

#     def compute_centroid_two_clusters(self, current_clusters, data_points_index):
#         size = len(data_points_index)
#         dim = self.dimension
#         centroid = [0.0]*dim
#         for index in data_points_index:
#             dim_data = current_clusters[str(index)]["centroid"]
#             for i in range(dim):
#                 centroid[i] += float(dim_data[i])
#         for i in range(dim):
#             centroid[i] /= size
#         return centroid

#     def compute_centroid(self, dataset, data_points_index):
#         size = len(data_points_index)
#         dim = self.dimension
#         centroid = [0.0]*dim
#         for idx in data_points_index:
#             dim_data = dataset[idx]["data"]
#             for i in range(dim):
#                 centroid[i] += float(dim_data[i])
#         for i in range(dim):
#             centroid[i] /= size
#         return centroid

#     def hierarchical_clustering(self):
#         """
#         Main Process for hierarchical clustering
#         """
#         dataset = self.dataset
#         current_clusters = self.clusters
#         old_clusters = []
#         heap = hc.compute_pairwise_distance(dataset)
#         heap = hc.build_priority_queue(heap)

#         while len(current_clusters) > self.k:
#             dist, min_item = heapq.heappop(heap)
#             # pair_dist = min_item[0]
#             pair_data = min_item[1]

#             # judge if include old cluster
#             if not self.valid_heap_node(min_item, old_clusters):
#                 continue

#             new_cluster = {}
#             new_cluster_elements = sum(pair_data, [])
#             new_cluster_cendroid = self.compute_centroid(dataset, new_cluster_elements)
#             new_cluster_elements.sort()
#             new_cluster.setdefault("centroid", new_cluster_cendroid)
#             new_cluster.setdefault("elements", new_cluster_elements)
#             for pair_item in pair_data:
#                 old_clusters.append(pair_item)
#                 del current_clusters[str(pair_item)]
#             self.add_heap_entry(heap, new_cluster, current_clusters)
#             current_clusters[str(new_cluster_elements)] = new_cluster
#         current_clusters.sort()
#         return current_clusters
            
#     def valid_heap_node(self, heap_node, old_clusters):
#         pair_dist = heap_node[0]
#         pair_data = heap_node[1]
#         for old_cluster in old_clusters:
#             if old_cluster in pair_data:
#                 return False
#         return True
            
#     def add_heap_entry(self, heap, new_cluster, current_clusters):
#         for ex_cluster in current_clusters.values():
#             new_heap_entry = []
#             dist = self.euclidean_distance(ex_cluster["centroid"], new_cluster["centroid"])
#             new_heap_entry.append(dist)
#             new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
#             heapq.heappush(heap, (dist, new_heap_entry))

#     def evaluate(self, current_clusters):
#         gold_standard = self.gold_standard
#         current_clustes_pairs = []

#         for (current_cluster_key, current_cluster_value) in current_clusters.items():
#             tmp = list(itertools.combinations(current_cluster_value["elements"], 2))
#             current_clustes_pairs.extend(tmp)
#         tp_fp = len(current_clustes_pairs)

#         gold_standard_pairs = []
#         for (gold_standard_key, gold_standard_value) in gold_standard.items():
#             tmp = list(itertools.combinations(gold_standard_value, 2))
#             gold_standard_pairs.extend(tmp)
#         tp_fn = len(gold_standard_pairs)

#         tp = 0.0
#         for ccp in current_clustes_pairs:
#             if ccp in gold_standard_pairs:
#                 tp += 1

#         if tp_fp == 0:
#             precision = 0.0
#         else:
#             precision = tp/tp_fp
#         if tp_fn == 0:
#             precision = 0.0
#         else:
#             recall = tp/tp_fn

#         return precision, recall

#     """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#     """                             Helper Functions                                 """
#     """                                                                              """    
#     """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#     def load_data(self, input_file_name):
#         """
#         load data and do some preparations
#         """
#         input_file = open(input_file_name, 'rU')
#         dataset = []
#         clusters = {}
#         gold_standard = {}
#         id = 0
#         for line in input_file:
#             line = line.strip('\n')
#             row = str(line)
#             row = row.split(",")
#             iris_class = row[-1]

#             data = {}
#             data.setdefault("id", id)   # duplicate
#             data.setdefault("data", row[:-1])
#             data.setdefault("class", row[-1])
#             dataset.append(data)

#             clusters_key = str([id])
#             clusters.setdefault(clusters_key, {})
#             clusters[clusters_key].setdefault("centroid", row[:-1])
#             clusters[clusters_key].setdefault("elements", [id])

#             gold_standard.setdefault(iris_class, [])
#             gold_standard[iris_class].append(id)

#             id += 1
#         return dataset, clusters, gold_standard

#     def quit(self, err_desc):
#         raise SystemExit('\n'+ "PROGRAM EXIT: " + err_desc + ', please check your input' + '\n')

#     def loaded_dataset(self):
#         """
#         use for test only
#         """
#         return self.dataset

#     def display(self, current_clusters, precision, recall):
#         print precision
#         print recall
#         clusters = current_clusters.values()
#         for cluster in clusters:
#             cluster["elements"].sort()
#             print cluster["elements"]