#!/usr/bin/env python3
"""
Decision Tree node counting module.
"""


class Node:
    """Represents an he tree."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        depth=0,
        is_root=False,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def count_nodes_below(self, only_leaves=False):
        """Counts all is node."""
        # Sol və sağ alt qollardbla
        left_count = (
            self.left_child.count_nodes_below(only_leaves=only_leaves)
            if self.left_child
            else 0
        )
        right_count = (
            self.right_child.count_nodes_below(only_leaves=only_leaves)
            if self.right_child
            else 0
        )

        # Əgər yalnız ə say (+1)
        current_node_count = 0 if only_leaves else 1

        return left_count + right_count + current_node_count


class Leaf:
    """Represents a leaf node."""

    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """A leaf node always counts as 1."""
        return 1


class Decision_Tree:
    """Represents the complete Decision Tree."""

    def __init__(self, root=None):
        self.root = root

    def depth(self):
        """Calculates the maximum depth."""

        def _max_depth(node):
            if node is None:
                return 0
            if node.is_leaf:
                return node.depth
            return max(_max_depth(node.left_child), _max_depth(node.right_child))

        return _max_depth(self.root)

    def count_nodes(self, only_leaves=False):
        """Counts the nodes in the decision tree."""
        if self.root is None:
            return 0
        return self.root.count_nodes_below(only_leaves=only_leaves)
