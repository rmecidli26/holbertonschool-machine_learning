#!/usr/bin/env python3
"""
Decision Tree module.
Provides classes for Tree Implementation.
"""
import numpy as np


def left_child_add_prefix(text):
    """Adds prefix for left child lines in the string representation."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    new_text = "+--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += "|  " + x + "\n"
    return new_text


def right_child_add_prefix(text):
    """Adds prefix for right child lines in the string representation."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    new_text = "+--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += "   " + x + "\n"
    return new_text


class Leaf:
    """Leaf class representing terminal nodes in the decision tree."""

    def __init__(self, value, depth=None):
        self.value = value
        self.depth = depth
        self.is_leaf = True
        self.sub_population = None

    def __str__(self):
        return f"-> leaf [value={self.value}]"

    def pred(self, x):
        return self.value

    def update_bounds_below(self):
        pass

    def get_leaves(self):
        return [self]

    def update_indicator(self):
        def indicator(A):
            mask = np.ones(A.shape[0], dtype=bool)
            for feature, value in self.lower.items():
                mask &= (A[:, feature] > value)
            for feature, value in self.upper.items():
                mask &= (A[:, feature] <= value)
            return mask
        self.indicator = indicator


class Node:
    """Node class representing internal decision nodes in the tree."""

    def __init__(
        self, feature=None, threshold=None, left_child=None,
        right_child=None, depth=0, is_root=False
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False
        self.sub_population = None

    def __str__(self):
        if self.is_root:
            node_str = (
                f"root [feature={self.feature}, "
                f"threshold={self.threshold}]\n"
            )
        else:
            node_str = (
                f"-> node [feature={self.feature}, "
                f"threshold={self.threshold}]\n"
            )

        left_str = ""
        if self.left_child:
            left_str = left_child_add_prefix(str(self.left_child))

        right_str = ""
        if self.right_child:
            right_str = right_child_add_prefix(str(self.right_child))

        return (node_str + left_str + right_str).rstrip('\n')

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

    def update_bounds_below(self):
        if self.is_root:
            if not hasattr(self, 'lower'):
                self.lower = {}
            if not hasattr(self, 'upper'):
                self.upper = {}

        for child in [self.left_child, self.right_child]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

        if self.left_child:
            self.left_child.lower[self.feature] = self.threshold
            self.left_child.update_bounds_below()

        if self.right_child:
            self.right_child.upper[self.feature] = self.threshold
            self.right_child.update_bounds_below()

    def get_leaves(self):
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves()
        if self.right_child:
            leaves += self.right_child.get_leaves()
        return leaves


class Decision_Tree:
    """Decision Tree class managing training and predictions."""

    def __init__(
        self, split_criterion="random", max_depth=100, min_pop=2, seed=0
    ):
        self.root = Node(is_root=True)
        self.split_criterion = split_criterion
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.rng = np.random.default_rng(seed)
        self.explanatory = None
        self.target = None
        self.predict = None

    def __str__(self):
        if self.root is None:
            return ""
        return str(self.root)

    def pred(self, x):
        """Predicts the value for an oom root."""
        if self.root:
            return self.root.pred(x)
        return None

    def np_extrema(self, arr):
        """Returns minimum and maximum of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly selects a feature and threshold for splitting."""
        diff = 0
        feature_min, feature_max = 0, 0
        feature = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            masked_arr = self.explanatory[:, feature][node.sub_population]
            if masked_arr.size == 0:
                break
            feature_min, feature_max = self.np_extrema(masked_arr)
            diff = feature_max - feature_min

        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """Trains the decision tree on the given data."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            # Gini criterion will be assigned here in later tasks
            pass

        self.explanatory = explanatory
        self.target = target

        # Initialize the root's sub population with all True
        self.root.sub_population = np.ones_like(self.target, dtype=bool)

        # Recursively build the tree
        self.fit_node(self.root)

        # Prepare lambda functions for vectorized predictions
        self.update_predict()

        if verbose == 1:
            print("  Training finished.")
            print(f"  - Depth                     : {self.depth()}")
            print(f"  - Number of nodes           : {self.count_nodes()}")
            print(f"  - Number of leaves          : {self.count_nodes(only_leaves=True)}")
            print(f"  - Accuracy on training data : {self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):
        """Recursively splits the node's population."""
        node.feature, node.threshold = self.split_criterion(node)

        # Vectorized split (no loops)
        left_mask = self.explanatory[:, node.feature] > node.threshold

        left_pop = node.sub_population & left_mask
        right_pop = node.sub_population & ~left_mask

        # Check leaf conditions
        def check_if_leaf(pop):
            if np.sum(pop) == 0:
                return True
            if np.sum(pop) < self.min_pop:
                return True
            if (node.depth + 1) >= self.max_depth:
                return True
            if np.unique(self.target[pop]).size == 1:
                return True
            return False

        # Build left subtree
        if check_if_leaf(left_pop):
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        # Build right subtree
        if check_if_leaf(right_pop):
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Creates a Leaf child node with the most frequent class."""
        if np.sum(sub_population) == 0:
            targets = self.target[node.sub_population]
        else:
            targets = self.target[sub_population]

        value = np.bincount(targets).argmax()

        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal Node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calculates accuracy over the given testing set."""
        preds = self.predict(test_explanatory)
        return np.sum(np.equal(preds, test_target)) / test_target.size

    def update_predict(self):
        """Updates boundary parameters and lambda for predictions."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves],
            axis=0
        )

    def update_bounds(self):
        if self.root:
            self.root.update_bounds_below()

    def get_leaves(self):
        if self.root:
            return self.root.get_leaves()
        return []

    def depth(self, node=None):
        """Calculates the maximum depth of the tree."""
        if node is None:
            node = self.root
        if getattr(node, 'is_leaf', False) or node is None:
            return getattr(node, 'depth', 0)

        l_d = self.depth(node.left_child) if node.left_child else 0
        r_d = self.depth(node.right_child) if node.right_child else 0
        return max(l_d, r_d)

    def count_nodes(self, node=None, only_leaves=False):
        """Counts the total number of nodes or only the leaves."""
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
