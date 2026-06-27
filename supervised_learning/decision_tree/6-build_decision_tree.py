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

    # --- Əvvəlki tapşırıqlardan gələn metodlar ---
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
    """ Node class representing internal nodes """
    {def __init__(self, feature=None, threshold=None, left_child=None, 
    right_child=None, depth=0, is_root=False):}
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

        left_str = left_child_add_prefix(str(self.left_child)) 
        if self.left_child else ""
        right_str = right_child_add_prefix(str(self.right_child)) 
        if self.right_child else ""

        full_str = node_str + left_str + right_str
        return full_str.rstrip('\n')

    def pred(self, x):
        """ Recursively finds the leaation """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

    # --- Əvvəlki tapşırıqlardan gələn metodlar ---
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
    """ Decision Tree class """
    def __init__(self, root=None):
        self.root = root

    def __str__(self):
        if self.root is None:
            return ""
        return str(self.root)

    def pred(self, x):
        """ Predicts the valoot """
        if self.root:
            return self.root.pred(x)
        return None

    def update_predict(self):
        """ Updates the  tree """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
            
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves], 
            axis=0
        )

    # --- Əvvəlki tapşırıqlardan gələn metodlar ---
    def update_bounds(self):
        if self.root:
            self.root.update_bounds_below()

    def get_leaves(self):
        if self.root:
            return self.root.get_leaves()
        return []
