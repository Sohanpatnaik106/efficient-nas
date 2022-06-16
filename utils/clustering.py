import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from src.model import BaseModel, _vgg

class HierarchicalClustering():

	def __init__(self, model_name, num_classes = 100, init_weights = True, device = "cpu",
					dropout = 0.5, batch_norm = True, weights = None, progress = True, 
					track_running_stats = False, dataloader = None, search_space = None, batch_size = 256,
					distance_type = "euclidean", linkage_type = "single", num_clusters = 5):


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
		
		self.distance_type = distance_type
		self.linkage_type = linkage_type
		self.num_clusters = num_clusters

		self.datasize = len(self.dataloader)
		self.num_configs = len(self.search_space)

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
		
		self.distance_matrix = np.tril(self.distance_matrix)
		self.distance_matrix[self.distance_matrix == 0] = np.inf
	
	def cluster(self):

		self.construct_distance_matrix()

		df = pd.DataFrame(data = np.ones(self.num_configs) * np.inf) #Initialized a dataframe which will store which point is in which cluster
		
		if self.num_clusters > self.distance_matrix.shape[0]: #If user provides impractical cut-off, cluster everthing into one cluster and not listen to user 
			self.num_clusters = self.distance_matrix.shape[0]


		if self.linkage_type == "single": #This 1 means formula of single linkage will be used, it is explained ahead
			
			d = {} #This dictionary keeps record of which data points or cluster are merging, hence can be used to make a dendogram
			
			for i in range(0, self.num_clusters):
			
				ij_min = np.unravel_index(self.distance_matrix.argmin(), self.distance_matrix.shape) #from the distance matrix, get the minimum distance
				#np.unravel_index gives us the position of minimum distance. e.g. (0,4) is where minimum value is present in matrix.
				#This is what we need as in Hierarchical clustering, we merge the two pairs with minimum distance
				if i == 0:
					df.iloc[ij_min[0]] = 0
					df.iloc[ij_min[1]] = 0
				else:
					try:
						a = int(df.iloc[ij_min[0]])
					except:
						df.iloc[ij_min[0]] = i
						a = i
					try:
						b = int(df.iloc[ij_min[1]])
					except:
						df.iloc[ij_min[1]] = i
						b = i
					df[(df[0]==a) | (df[0]==b)] = i
				d[i] = ij_min
				#The if, else code till here is just filling the dataframe as the two points/clusters combine.
				#So, for example if 1 and 2 combines, dataframe will have 1 : 0, 2 : 0. Which means point 1 and 2 both are in same cluster (0th cluster)
				for j in range(0, ij_min[0]):
					#we want to ignore the diagonal, and diagonal is 0. We replaced 0 by infinte. 
					#So this if condition will skip diagonals
					if np.isfinite(self.distance_matrix[ij_min[0]][j]) and np.isfinite(self.distance_matrix[ij_min[1]][j]):
						#after two points/cluster are linked, to calculate new distance we take minimum distance for single linkage
						self.distance_matrix[ij_min[1]][j] = min(self.distance_matrix[ij_min[0]][j], self.distance_matrix[ij_min[1]][j])
				# To avoid the combined data points/cluster in further calculations, we make them infinte.
				#Our if loop above this, will therefore skip the infinite record entries.
				self.distance_matrix[ij_min[0]] = np.inf
			
			return d, df
			# return d, df[0].as_matrix()
		
		elif self.linkage_type == "complete":
			d_complete = {}
			for i in range(0,self.num_clusters):
				ij_min = np.unravel_index(self.distance_matrix.argmin(), self.distance_matrix.shape)
				if i == 0:
					df.iloc[ij_min[0]] = 0
					df.iloc[ij_min[1]] = 0
				else:
					try:
						a = int(df.iloc[ij_min[0]])
					except:
						df.iloc[ij_min[0]] = i
						a = i
					try:
						b = int(df.iloc[ij_min[1]])
					except:
						df.iloc[ij_min[1]] = i
						b = i
					df[(df[0]==a) | (df[0]==b)] = i
				d_complete[i] = ij_min
				for j in range(0, ij_min[0]):
					if np.isfinite(self.distance_matrix[ij_min[0]][j]) and np.isfinite(self.distance_matrix[ij_min[1]][j]):
						#after two points/cluster are linked, to calculate new distance we take maximum distance for complete linkage
						self.distance_matrix[ij_min[1]][j] = max(self.distance_matrix[ij_min[0]][j], self.distance_matrix[ij_min[1]][j])
				self.distance_matrix[ij_min[0]] = np.inf
			return d_complete, df
			# return d_complete, df[0].as_matrix()
		
		elif self.linkage_type == "average":

			d_average = {}
			for i in range(0,self.num_clusters):
				ij_min = np.unravel_index(self.distance_matrix.argmin(), self.distance_matrix.shape)
				if i == 0:
					df.iloc[ij_min[0]] = 0
					df.iloc[ij_min[1]] = 0
				else:
					try:
						a = int(df.iloc[ij_min[0]])
					except:
						df.iloc[ij_min[0]] = i
						a = i
					try:
						b = int(df.iloc[ij_min[1]])
					except:
						df.iloc[ij_min[1]] = i
						b = i
					df[(df[0]==a) | (df[0]==b)] = i
				d_average[i] = ij_min
				for j in range(0, ij_min[0]):
					if np.isfinite(self.distance_matrix[ij_min[0]][j]) and np.isfinite(self.distance_matrix[ij_min[1]][j]):
						#after two points/cluster are linked, to calculate new distance we take average distance for average linkage
						self.distance_matrix[ij_min[1]][j] = (self.distance_matrix[ij_min[0]][j] + self.distance_matrix[ij_min[1]][j])/2.0          
				self.distance_matrix[ij_min[0]] = np.inf
			return d_average, df
			# return d_average, df[0].as_matrix()

		