#!/usr/bin/env python3
import sys
sys.setrecursionlimit(100000)
def summation_i_squared(n):
    if type(n) is not int or n < 0:
        return None
    if n == 0:
        return 0
    if n == 1:
        return 1
    return (n * (n + 1) * (2 * n + 1)) // 6
