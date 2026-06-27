#!/usr/bin/env python3
"""
Decision Tree builder module.
Contains basic classes Leaf, Node, and Decision_Tree.
"""


class Leaf:
    """Leaf node of a decision tree."""

    def __init__(self, value, depth=None):
        """Initializes a leaf node with a value and an optional depth."""
        self.value = value
        self.depth = depth
        # Yarpaq olduğunu bildirən atribut
        self.is_leaf = True


class Node:
    """Internal node of a decision tree."""

    def __init__(
        self,
        feature=0,
        threshold=0,
        left_child=None,
        right_child=None,
        depth=0,
        is_root=False
    ):
        """Initializes an internal node for the decision tree."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        # Daxili düyün olduğu üçün False təyin edilir
        self.is_leaf = False


class Decision_Tree:
    """Decision Tree model class."""

    def __init__(self, root=None):
        """Initializes the decision tree with a root node."""
        self.root = root
