#!/usr/bin/env python3
def summation_i_squared(n):
    if type(n) is not int:
        return None
    if n < 0:
        return None
    if n == 0:
        return 0
    return (n * (n + 1) * (2 * n + 1)) // 6
