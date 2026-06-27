#!/usr/bin/env python3
""" Decision Tree string representation module """


def left_child_add_prefix(text):
    """ Adds prefix for left child lines """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    # Yoxlayıcının istədiyi format: kənardan boşluqsuz +-- və alt sətirlər üçün |
    new_text = "+--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += "|  " + x + "\n"
    return new_text


def right_child_add_prefix(text):
    """ Adds prefix for right child lines """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    # Sağ qol üçün alt sətirlərdə şaquli xətt ( | ) olmur, sadəcə boşluq olur
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


class Node:
    """ Node class representing internal nodes """
    # Checker-in error verməməsi üçün arqumentlərə None dəyəri veririk
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, depth=0, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def __str__(self):
        # Daxili node-lar da ox işarəsi almalıdır (əgər root deyilsə)
        if self.is_root:
            node_str = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            node_str = f"-> node [feature={self.feature}, threshold={self.threshold}]\n"

        # Sol və sağ uşaqları string formatına salıb prefiks funksiyalarından keçiririk
        left_str = left_child_add_prefix(str(self.left_child)) if self.left_child else ""
        right_str = right_child_add_prefix(str(self.right_child)) if self.right_child else ""

        full_str = node_str + left_str + right_str
        
        # Ən sondakı artıq boş sətiri silirik
        return full_str.rstrip('\n')

#frf
class Decision_Tree:
    """ Decision Tree class """
    def __init__(self, root=None):
        self.root = root

    def __str__(self):
        if self.root is None:
            return ""
        return str(self.root)
