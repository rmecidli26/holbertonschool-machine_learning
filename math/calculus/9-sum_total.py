#!/usr/bin/env python3

def summation_i_squared(n):
    if not isinstance(n, int):
        return None
    if n < 1:
        return 0
    result = (n * (n + 1) * (2 * n + 1)) // 6
    return result
