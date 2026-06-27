#!/usr/bin/env python3
"""
Decision Tree builder module.
Provides classes to build, manage bounds, and compute indicators.
"""
import numpy as np


class Leaf:
    """Leaf node of a decision tree."""

    def __init__(self, value, depth=None):
        """Initializes a leaf node."""
        self.value = value
        self.depth = depth
        self.is_leaf = True
        self.lower = {}
        self.upper = {}
        self.indicator = None

    def __str__(self):
        """Returns string representation of the leaf."""
        return f"-> leaf [value={self.value}]"

    def get_leaves(self):
        """Returns a list containing this leaf."""
        return [self]

    def update_bounds_below(self):
        """Updates bounds below this node (pass for leaf)."""
        pass

    def update_indicator(self):
        """Computes and stores the indicator function."""
        def is_large_enough(x):
            keys = list(self.lower.keys())
            if not keys:
                return np.ones(x.shape[0], dtype=bool)
            conds = np.array([
                np.greater(x[:, k], self.lower[k]) for k in keys
            ])
            return np.all(conds, axis=0)

        def is_small_enough(x):
            keys = list(self.upper.keys())
            if not keys:
                return np.ones(x.shape[0], dtype=bool)
            conds = np.array([
                np.less_equal(x[:, k], self.upper[k]) for k in keys
            ])
            return np.all(conds, axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0
        )


class Node:
    """Internal node of a decision tree."""

    def __init__(
        self, feature=0, threshold=0, left_child=None,
        right_child=None, depth=0, is_root=False
    ):
        """Initializes an internal node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False
        self.lower = {}
        self.upper = {}
        self.indicator = None

    def left_child_add_prefix(self, text):
        """Adds prefix for left child string."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds prefix for right child string."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Returns string representation of the node."""
        if self.is_root:
            res = (f"root [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        else:
            res = (f"-> node [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")

        if self.left_child:
            res += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            res += self.right_child_add_prefix(str(self.right_child))

        return res.rstrip()

    def get_leaves(self):
        """Returns all leaves under this node."""
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves()
        if self.right_child:
            leaves += self.right_child.get_leaves()
        return leaves

    def update_bounds_below(self):
        """Updates lower and upper bounds for children."""
        if self.is_root:
            self.upper = {}
            self.lower = {}

        for child, is_left in [
            (self.left_child, True), (self.right_child, False)
        ]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                
                # Sol övlad (Left child): x > threshold olduğu üçün LOWER yenilənir
                if is_left:
                    child.lower[self.feature] = max(
                        child.lower.get(self.feature, -np.inf),
                        self.threshold
                    )
                # Sağ övlad (Right child): x <= threshold olduğu üçün UPPER yenilənir
                else:
                    child.upper[self.feature] = min(
                        child.upper.get(self.feature, np.inf),
                        self.threshold
                    )
                child.update_bounds_below()

    def update_indicator(self):
        """Computes and stores the indicator function."""
        def is_large_enough(x):
            keys = list(self.lower.keys())
            if not keys:
                return np.ones(x.shape[0], dtype=bool)
            conds = np.array([
                np.greater(x[:, k], self.lower[k]) for k in keys
            ])
            return np.all(conds, axis=0)

        def is_small_enough(x):
            keys = list(self.upper.keys())
            if not keys:
                return
