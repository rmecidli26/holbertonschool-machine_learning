#!/usr/bin/env python3
""" Decision Tree string representation module """


def left_child_add_prefix(text):
    """ Adds prefix for left child lines """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    new_text = "    +---" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("    |   " + x) + "\n"
    return new_text


def right_child_add_prefix(text):
    """ Adds prefix for right child lines """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    new_text = "    +---" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("        " + x) + "\n"
    return new_text


class Leaf:
    """ Leaf class representing terminal nodes """
    def __init__(self, value, depth=None):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        return f"-> leaf [value={self.value}]"


class Node:
    """ Node class representing internal nodes """
    def __init__(self, feature, threshold, left_child, right_child, depth=None, is_root=False):
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
            node_str = f"node [feature={self.feature}, threshold={self.threshold}]\n"

        left_str = left_child_add_prefix(self.left_child.__str__())
        right_str = right_child_add_prefix(self.right_child.__str__())

        full_str = node_str + left_str + right_str
        if self.is_root:
            return full_str.rstrip('\n')
        return full_str


class Decision_Tree:
    """ Decision Tree class """
    def __init__(self, root=None):
        self.root = root

    def __str__(self):
        if self.root is None:
            return ""
        return self.root.__str__()
