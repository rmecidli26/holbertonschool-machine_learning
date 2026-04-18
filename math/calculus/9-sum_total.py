#!/usr/bin/env python3
def summation_i_squared(n):
    if not n.isdigit():
        return None
    if n <= 0:
        return 0
    else:
        return n*n + summation_i_squared(n-1)
