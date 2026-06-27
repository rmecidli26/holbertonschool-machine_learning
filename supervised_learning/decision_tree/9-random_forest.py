#!/usr/bin/env python3
"""
Random Forest module.
Provides Random_Forest class using a collection of Decision Trees.
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """Random Forest class implementing ensemble learning."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initializes the Random Forest model."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predicts the target for the given explanatory data."""
        # Initialize an empty list to store predictions from individual trees
        tree_predictions = []

        # Generate predictions for each tree in the forest
        for pred_func in self.numpy_preds:
            tree_predictions.append(pred_func(explanatory))

        # Convert to numpy array for easier column-wise operations
        tree_predictions = np.array(tree_predictions)

        # Calculate the mode (most frequent) prediction for each example
        modes = []
        for i in range(tree_predictions.shape[1]):
            # Get all predictions for the i-th individual
            individual_preds = tree_predictions[:, i].astype(int)
            # Find the most frequent class (mode) using bincount
            most_frequent = np.bincount(individual_preds).argmax()
            modes.append(most_frequent)

        return np.array(modes)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Fits the random forest to the training data."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            mean_d = np.array(depths).mean()
            mean_n = np.array(nodes).mean()
            mean_l = np.array(leaves).mean()
            mean_a = np.array(accuracies).mean()
            forest_a = self.accuracy(self.explanatory, self.target)

            print("  Training finished.")
            print(f"    - Mean depth                     : {mean_d}")
            print(f"    - Mean number of nodes           : {mean_n}")
            print(f"    - Mean number of leaves          : {mean_l}")
            print(f"    - Mean accuracy on training data : {mean_a}")
            print(f"    - Accuracy of the forest on td   : {forest_a}")

    def accuracy(self, test_explanatory, test_target):
        """Calculates the accuracy of the forest on test data."""
        predictions = self.predict(test_explanatory)
        correct_preds = np.equal(predictions, test_target)
        return np.sum(correct_preds) / test_target.size
