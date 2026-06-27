#!/usr/bin/env python3
"""
Isolation Random Forest module.
Provides a class for building an ensemble of isolation trees to
detect outliers based on mean prediction depths.
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """Isolation Random Forest class for anomaly detection."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initializes the Isolation Random Forest model."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Returns the mean depth prediction for each individual."""
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Trains the isolation random forest on the explanatory data."""
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []

        for i in range(n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth,
                seed=self.seed + i
            )
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))

        if verbose == 1:
            mean_d = np.array(depths).mean()
            mean_n = np.array(nodes).mean()
            mean_l = np.array(leaves).mean()

            print("  Training finished.")
            print(f"    - Mean depth                     : {mean_d}")
            print(f"    - Mean number of nodes           : {mean_n}")
            print(f"    - Mean number of leaves          : {mean_l}")

    def suspects(self, explanatory, n_suspects):
        """
        Returns the n_suspects rows in explanatory that have the
        smallest mean depth, indicating they are likely outliers.
        """
        depths = self.predict(explanatory)
        
        # Ən kiçik dərinliklərin indekslərini tapırıq
        suspect_indices = np.argsort(depths)[:n_suspects]
        
        # Həmin indekslərə uyğun gələn sətirləri və dərinlikləri qaytarırıq
        suspect_rows = explanatory[suspect_indices]
        suspect_depths = depths[suspect_indices]
        
        return suspect_rows, suspect_depths
