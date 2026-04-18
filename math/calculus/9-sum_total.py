#!/usr/bin/env python3
def summation_i_squared(n):
    if not isinstance(n, int) or n < 0:
        return None
    res = (n * (n + 1) * (2 * n + 1)) / 6
    return int(res)
