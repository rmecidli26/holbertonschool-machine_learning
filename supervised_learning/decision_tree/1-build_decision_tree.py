#!/usr/bin/env python3
"""Decision Tree"""


class Node:
    """Internal node"""

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
        """Count nodes"""
        lc = self.left_child
        rc = self.right_child
        left = lc.count_nodes_below(only_leaves) if lc else 0
        right = rc.count_nodes_below(only_leaves) if rc else 0
        return left + right + (0 if only_leaves else 1)


class Leaf:
    """Leaf node"""

    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """Count leaf"""
        return 1


class Decision_Tree:
    """Tree class"""

    def __init__(self, root=None):
        self.root = root

    def depth(self):
        """Calculate depth"""

        def _max_depth(node):
            if node is None:
                return 0
            if node.is_leaf:
                return node.depth
            return max(
                _max_depth(node.left_child), _max_depth(node.right_child)
            )

        return _max_depth(self.root)

    def count_nodes(self, only_leaves=False):
        """Count total"""
        if self.root is None:
            return 0
        return self.root.count_nodes_below(only_leaves=only_leaves)
