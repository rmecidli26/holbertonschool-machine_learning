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
        lc, rc = self.left_child, self.right_child
        left = lc.count_nodes_below(only_leaves) if lc else 0
        right = rc.count_nodes_below(only_leaves) if rc else 0
        return left + right + (0 if only_leaves else 1)

    def left_child_add_prefix(self, text):
        """Left prefix"""
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if not lines:
            return ""
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |     " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Right prefix"""
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if not lines:
            return ""
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "          " + x + "\n"
        return new_text

    def __str__(self):
        """String format"""
        t = "root" if self.is_root else "node"
        string = f"{t} [feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            string += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            string += self.right_child_add_prefix(str(self.right_child))
        return string


class Leaf:
    """Leaf node"""

    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """Count leaf"""
        return 1

    def __str__(self):
        """String format"""
        return f"leaf [value={self.value}]\n"


class Decision_Tree:
    """Tree class"""

    def __init__(self, root=None):
        self.root = root

    def depth(self):
        """Depth check"""

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

    def __str__(self):
        """String format"""
        if self.root is None:
            return ""
        return str(self.root).rstrip("\n")
