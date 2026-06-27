#!/usr/bin/env python3
"""Decision Tree"""


def left_child_add_prefix(text):
    lines = text.split("\n")
    # Əgər sonuncu sətnzərə alm
    if lines[-1] == "":
        lines.pop()
    new_text = "    +---" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("    |   " + x) + "\n"
    return new_text

def right_child_add_prefix(text):
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    new_text = "    +---" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("        " + x) + "\n"
    return new_text
