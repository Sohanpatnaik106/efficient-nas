import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, ward

# num_configs is search_space.shape[0]
def plot_sampling_prob_dist(sampling_probabilities, epoch_number, num_configs, visualisation_dir, normalise = True, temperature = None):

    if normalise:
        sampling_probabilities = np.exp(sampling_probabilities / temperature) \
                                            / np.sum(np.exp(sampling_probabilities / temperature))
        

    plt.figure(figsize = (10, 5))
    plt.title(f"Sampling Probability Distribution after {epoch_number} epoch(s)")
    architecture_indices = np.arange(num_configs)
    
    plt.bar(architecture_indices, sampling_probabilities, color = "blue")
    plt.xlabel("Architecture Indices")
    plt.ylabel("Sampling Probabilities")
    plt.savefig(os.path.join(visualisation_dir, f"{epoch_number}.png"))
    plt.clf()

def plot_loss(training_loss, validation_loss, test_loss, visualisation_dir, num_epochs = 180):

    plt.figure(figsize = (10, 5))
    plt.title(f"Variation of loss with respect to number of epochs")
    epoch_indices = np.arange(1, num_epochs+1)

    plt.plot(epoch_indices, training_loss, label = "Training Loss")
    plt.plot(epoch_indices, validation_loss, label = "Validation Loss")
    plt.plot(epoch_indices, test_loss, label = "Test Loss")
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(visualisation_dir, "loss_plot.png"))
    plt.clf()

def plot_accuracy(training_accuracy, validation_accuracy, test_accuracy, visualisation_dir, num_epochs = 180):

    plt.figure(figsize = (10, 5))
    plt.title(f"Variation of accuracy with respect to number of epochs")
    epoch_indices = np.arange(1, num_epochs+1)

    plt.plot(epoch_indices, training_accuracy, label = "Training Accuracy")
    plt.plot(epoch_indices, validation_accuracy, label = "Validation Accuracy")
    plt.plot(epoch_indices, test_accuracy, label = "Test Accuracy")
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(visualisation_dir, "accuracy_plot.png"))
    plt.clf()

def plot_hierarchy_dendogram(hierarchy_tree, labels, visualisation_dir):

    plt.figure(figsize=(25,10))
    plt.title("Hierarchical Clustering of Model Architectures")
    plt.xlabel("Model Index")
    plt.ylabel("Distance")
    dendrogram(
        hierarchy_tree,
        leaf_rotation = 90.,
        leaf_font_size = 8.,
        labels = labels,
    )
    plt.savefig(os.path.join(visualisation_dir, "dendrogram.png"), bbox_inches = 'tight')
    plt.clf()

def plot_hierarchical_sampling_prob_dist(cluster_nodelist, epoch_number, num_configs, visualisation_dir, normalise = True, temperature = None):

    sampling_probabilities = np.zeros(num_configs)
    for node in cluster_nodelist:
        if node.get_id() < num_configs:
            sampling_probabilities[node.get_id()] = node.get_sample_probability()

    if normalise:
        sampling_probabilities = np.exp(sampling_probabilities / temperature) \
                                            / np.sum(np.exp(sampling_probabilities / temperature))
        

    plt.figure(figsize = (10, 5))
    plt.title(f"Sampling Probability Distribution after {epoch_number} epoch(s)")
    architecture_indices = np.arange(num_configs)
    
    plt.bar(architecture_indices, sampling_probabilities, color = "blue")
    plt.xlabel("Architecture Indices")
    plt.ylabel("Sampling Probabilities")
    plt.savefig(os.path.join(visualisation_dir, f"{epoch_number}.png"))
    plt.clf()