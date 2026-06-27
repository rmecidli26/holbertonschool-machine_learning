#!/usr/bin/env python3
"""
Isolation Random Tree module.
Provides an Isolation_Random_Tree class for outlier detection.
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """Isolation tree class for building random trees to find outliers."""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initializes the Isolation Random Tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Returns string representation of the tree starting from root."""
        if self.root is None:
            return ""
        return str(self.root)

    def depth(self, node=None):
        """Calculates and returns the maximum depth of the tree."""
        if node is None:
            node = self.root
        if getattr(node, 'is_leaf', False) or node is None:
            return getattr(node, 'depth', 0)

        l_d = self.depth(node.left_child) if node.left_child else 0
        r_d = self.depth(node.right_child) if node.right_child else 0
        return max(l_d, r_d)

    def count_nodes(self, node=None, only_leaves=False):
        """Counts and returns the total number of nodes (or only leaves)."""
        if node is None:
            node = self.root
        if node is None:
            return 0

        if getattr(node, 'is_leaf', False):
            return 1

        count = 0 if only_leaves else 1
        if node.left_child:
            count += self.count_nodes(node.left_child, only_leaves)
        if node.right_child:
            count += self.count_nodes(node.right_child, only_leaves)

        return count

    def update_bounds(self):
        """Initiates updating spatial bounds from the root."""
        if self.root:
            self.root.update_bounds_below()

    def get_leaves(self):
        """Returns all leaves in the isolation tree."""
        if self.root:
            return self.root.get_leaves()
        return []

    def update_predict(self):
        """Updates the internal prediction function lambda."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves],
            axis=0
        )

    def np_extrema(self, arr):
        """Returns the minimum and maximum of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly chooses a feature and a threshold to split the node."""
        diff = 0
        feature_min, feature_max = 0, 0
        feature = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            masked = self.explanatory[:, feature][node.sub_population]
            if masked.size == 0:
                break
            feature_min, feature_max = self.np_extrema(masked)
            diff = feature_max - feature_min

        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf child returning the depth as its value."""
        # For isolation trees, prediction value is the depth of the leaf
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates and returns a new internal node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively fits the node by splitting the population."""
        node.feature, node.threshold = self.random_split_criterion(node)

        mask = self.explanatory[:, node.feature] > node.threshold
        left_population = node.sub_population & mask
        right_population = node.sub_population & ~mask

        # Check if left node should be a leaf (size <= min_pop or max depth)
        is_left_leaf = (
            np.sum(left_population) <= self.min_pop or
            (node.depth + 1) >= self.max_depth
        )

        if is_left_leaf:
            child = self.get_leaf_child(node, left_population)
            node.left_child = child
        else:
            child = self.get_node_child(node, left_population)
            node.left_child = child
            self.fit_node(node.left_child)

        # Check if right node should be a leaf
        is_right_leaf = (
            np.sum(right_population) <= self.min_pop or
            (node.depth + 1) >= self.max_depth
        )

        if is_right_leaf:
            child = self.get_leaf_child(node, right_population)
            node.right_child = child
        else:
            child = self.get_node_child(node, right_population)
            node.right_child = child
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Trains the isolation random tree on given explanatory data."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        n_samples = explanatory.shape[0]
        self.root.sub_population = np.ones(n_samples, dtype=bool)

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            leaf_cnt = self.count_nodes(only_leaves=True)
            print("  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : {leaf_cnt}")
