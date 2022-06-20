import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from src.model import BaseModel, _vgg
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, ward
from scipy.spatial.distance import squareform
import numpy as np 
import treelib
from .visualise import plot_hierarchy_dendogram

class HierarchicalClustering():

	def __init__(self, model_name, num_classes = 100, init_weights = True, device = "cpu",
					dropout = 0.5, batch_norm = True, weights = None, progress = True, 
					track_running_stats = False, dataloader = None, search_space = None, batch_size = 256,
					distance_type = "euclidean", linkage_type = "single", num_clusters = 5,
					visualisation_dir = None):


		self.model_name = model_name
		self.num_classes = num_classes
		self.init_weights = init_weights
		self.device = device
		self.dropout = dropout
		self.batch_norm = batch_norm
		self.weights = weights
		self.progress = progress
		self.track_running_stats = track_running_stats
		self.dataloader = dataloader
		self.search_space = search_space
		self.batch_size = batch_size
		self.visualisation_dir = visualisation_dir
		
		self.distance_type = distance_type
		self.linkage_type = linkage_type
		self.num_clusters = num_clusters

		self.datasize = len(self.dataloader)
		self.num_configs = len(self.search_space)
		self.labels = list(self.search_space.keys())

		self.distance_matrix = np.zeros((self.num_configs, self.num_configs), dtype = np.float32)

	def create_model(self, model_config):

		# TODO: Only implemented for VGG architecture, make it generalised
		# new_model = _vgg(self.search_space[str(sample_idx)], self.weights, batch_norm = self.batch_norm, progress = self.progress)
		# new_model.load_state_dict(model.state_dict)

		model = BaseModel(self.model_name, model_config, num_classes = self.num_classes, init_weights = self.init_weights, 
							dropout = self.dropout, batch_norm = self.batch_norm, weights = self.weights, progress = self.progress, 
							track_running_stats = self.track_running_stats)
		
		return model

	# TODO: Implement this type of distance
	def manhattan_distance(self, vectorA, vectorB):
		raise NotImplementedError

	def euclidean_distance(self, vectorA, vectorB):
		return torch.sum(torch.sum((vectorA - vectorB).pow(2), dim = 1).sqrt())

	def compute_pairwise_distance(self, configA, configB):

		modelA = self.create_model(configA).to(self.device)
		modelB = self.create_model(configB).to(self.device)

		distance = 0

		with tqdm(self.dataloader, unit = "batch", position = 0, leave = True) as tepoch:
			for i, (images, labels) in enumerate(tepoch):
				tepoch.set_description(f"Hierarchical Clustering")
				
				outputsA = modelA.feature_forward(images.to(self.device))
				outputsB = modelB.feature_forward(images.to(self.device))

				if self.distance_type == "euclidean":
					distance += self.euclidean_distance(outputsA.detach(), outputsB.detach())

			distance /= self.datasize

		del modelA
		del modelB

		return distance
	
	def construct_distance_matrix(self):

		for i, (idxA, configA) in enumerate(self.search_space.items()):
			for j, (idxB, configB) in enumerate(self.search_space.items()):

				if i >= j:
					continue

				distance = self.compute_pairwise_distance(configA, configB)
				self.distance_matrix[i][j] = distance
				self.distance_matrix[j][i] = distance
		
		self.distance_matrix = squareform(self.distance_matrix)
	
	def cluster(self):

		self.construct_distance_matrix()

		hierarchy_tree = linkage(self.distance_matrix, self.linkage_type)
		plot_hierarchy_dendogram(hierarchy_tree, self.labels, self.visualisation_dir)
		return hierarchy_tree