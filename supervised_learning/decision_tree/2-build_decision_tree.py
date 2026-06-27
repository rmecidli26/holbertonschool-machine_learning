#!/usr/bin/env python3
"""
Decision Tree builder module.
Contains classes Leaf, Node, and Decision_Tree with string representations.
"""


class Leaf:
    """Leaf node of a decision tree."""

    def __init__(self, value, depth=None):
        """Initializes a leaf node with a value and an optional depth."""
        self.value = value
        self.depth = depth

    def __str__(self):
        """Returns the string representation of the leaf."""
        return f"-> leaf [value={self.value}]"


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

    def left_child_add_prefix(self, text):
        """Adds formatting prefix for the left child's string output."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Adds formatting prefix for the right child's string output."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Returns the string representation of the node and its children."""
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

        # Sağdakı əlavə boşluq və yeni sətirləri təmizləyirik
        return res.rstrip()


class Decision_Tree:
    """Decision Tree model class."""

    def __init__(self, root=None):
        """Initializes the decision tree with a root node."""
        self.root = root

    def __str__(self):
        """Returns the string representation of the entire tree."""
        if self.root is None:
            return ""
        return self.root.__str__()
