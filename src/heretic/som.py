# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import numpy as np
from minisom import MiniSom
from collections import defaultdict

class SOMCalculator:
    """
    A simplified class to train a Self-Organizing Map (SOM) and extract neuron weights.
    """
    def __init__(self, som_x, som_y, iterations, lr, sigma):
        """
        Initializes the SOM calculator with training parameters.

        Args:
            som_x (int): Number of neurons in the x-axis of the SOM grid.
            som_y (int): Number of neurons in the y-axis of the SOM grid.
            iterations (int): Number of training iterations for the SOM.
            lr (float): Learning rate for the SOM.
            sigma (float): Radius of the neighborhood function.
        """
        self.som_x = som_x
        self.som_y = som_y
        self.iterations = iterations
        self.lr = lr
        self.sigma = sigma
        self.som = None

    def fit(self, data: np.ndarray):
        """
        Trains the SOM on the provided 2D data.

        Args:
            data (np.ndarray): A 2D NumPy array of shape (n_samples, n_features).
        """
        # Ensure data is 2D
        if len(data.shape) != 2:
            raise ValueError(f"Data must be a 2D array, but got shape {data.shape}")

        n_samples, n_features = data.shape

        # Initialize and train the SOM using MiniSom
        self.som = MiniSom(
            self.som_x,
            self.som_y,
            n_features,
            sigma=self.sigma,
            learning_rate=self.lr,
            random_seed=0,  # For reproducibility
            activation_distance='euclidean',
            topology='hexagonal'
        )
        self.som.random_weights_init(data)
        self.som.train_random(data, self.iterations)

    def get_top_k_neuron_weights(self, k: int):
        """
        Gets the weights of the top-k neurons based on their frequency of being winners.

        Args:
            k (int): The number of top neurons to return.

        Returns:
            np.ndarray: A 2D array of shape (k, n_features) containing the weights of the top-k neurons.
        """
        if self.som is None:
            raise RuntimeError("SOM has not been trained yet. Call `fit()` first.")

        winners = np.array([self.som.winner(x) for x in self.som._weights.reshape(-1, self.som._weights.shape[2])])
        counts = defaultdict(int)
        for w in winners:
            counts[tuple(w)] += 1

        # Sort neurons by their count (descending) and get the top-k
        sorted_neurons = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:k]

        # Get the coordinates of the top-k neurons
        top_k_coords = [coord for coord, _ in sorted_neurons]

        # Fetch the weights for these top-k neurons
        # self.som.get_weights() has shape (som_x, som_y, n_features)
        top_k_weights = np.array([self.som.get_weights()[i, j] for i, j in top_k_coords])

        return top_k_weights
