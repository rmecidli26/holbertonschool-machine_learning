#!/usr/bin/env python3
""" Decision Tree module """
import numpy as np


def left_child_add_prefix(text):
    """ Adds prefix for left child lines """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    new_text = "+--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += "|  " + x + "\n"
    return new_text


def right_child_add_prefix(text):
    """ Adds prefix for right child lines """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    new_text = "+--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += "   " + x + "\n"
    return new_text


class Leaf:
    """ Leaf class representing terminal nodes """
    def __init__(self, value, depth=None):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        return f"-> leaf [value={self.value}]"

    def pred(self, x):
        """ Predicts the value for a single observation """
        return self.value

    # ========================================================
    # DİQQƏT: 3, 4 və 5-ci tapşırıqlardan olan metodlarınızı
    # (məsələn: indicator, update_indicator) BURAYA ƏLAVƏ EDİN!
    # ========================================================
    # def indicator(self, A):
    #     ...
    # def update_indicator(self):
    #     ...


class Node:
    """ Node class representing internal nodes """
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, depth=0, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def __str__(self):
        if self.is_root:
            node_str = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            node_str = f"-> node [feature={self.feature}, threshold={self.threshold}]\n"

        left_str = left_child_add_prefix(str(self.left_child)) if self.left_child else ""
        right_str = right_child_add_prefix(str(self.right_child)) if self.right_child else ""

        full_str = node_str + left_str + right_str
        return full_str.rstrip('\n')

    def pred(self, x):
        """ Recursively finds the leaf value for a single observation """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Decision_Tree:
    """ Decision Tree class """
    def __init__(self, root=None):
        self.root = root

    def __str__(self):
        if self.root is None:
            return ""
        return str(self.root)

    def pred(self, x):
        """ Predicts the value for a single observation using the root """
        return self.root.pred(x)

    def update_predict(self):
        """ Updates the prediction function for the entire tree """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
            
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves], 
            axis=0
        )

    # ========================================================
    # DİQQƏT: 3, 4 və 5-ci tapşırıqlardan olan metodlarınızı
    # (məsələn: update_bounds, get_leaves) BURAYA ƏLAVƏ EDİN!
    # ========================================================
    # def update_bounds(self):
    #     ...
    # def get_leaves(self):
    #     ...