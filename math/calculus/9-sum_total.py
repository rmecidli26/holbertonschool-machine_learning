#!/usr/bin/env python3
def summation_i_squared(n):
    try:
        if type(n) is not int or n < 0:
            return None
        if n == 0:
            return 0
        return (n * (n + 1) * (2 * n + 1)) // 6
    except (TypeError, NameError, AttributeError):
        return None
try:
    val = n_variable_not_defined
except NameError:
    val = "invalid"
print(summation_i_squared(val))
