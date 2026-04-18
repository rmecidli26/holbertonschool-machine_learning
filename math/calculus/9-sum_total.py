#!/usr/bin/env python3
def summation_i_squared(n):
    if not n.isdigit():
        return None
    sum = 0
    i = 0
    if(i < n):
        i = i + 1
        sum = sum + i*i
        return summation_i_squared(n):
