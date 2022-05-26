import numpy as np
import matplotlib.pyplot as plt

# num_configs is search_space.shape[0]
def plot_sampling_prob_dist(sampling_probabilities, epoch_number, num_configs, visualisation_dir):

    plt.figure((10, 5))
    plt.title(f"Sampling Probability Distribution after {epoch_number} epoch(s)")
    architecture_indices = np.arange(num_configs)
    
    plt.bar(architecture_indices, sampling_probabilities, color = "blue")
    plt.xlabel("Architecture Indices")
    plt.ylabel("Sampling Probabilities")
    plt.save()