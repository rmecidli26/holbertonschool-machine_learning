#!/usr/bin/env python3
"""
Decisres
"""


class Node:
    """Representree."""

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


class Leaf:
    """Represeion value."""

    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.is_leaf = True


class Decision_Tree:
    """Represents the complete Decision Tree model."""

    def __init__(self, root=None):
        self.root = root

    def depth(self):
        """Calculates the m."""

        def _max_depth(node):
            if node is None:
                return 0
            if node.is_leaf:
                return node.depth

            # Sol və saq
            left_depth = _max_depth(node.left_child)
            right_depth = _max_depth(node.right_child)

            return max(left_depth, right_depth)

        return _max_depth(self.root)
