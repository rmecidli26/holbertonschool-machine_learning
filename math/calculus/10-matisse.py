#!/usr/bin/env python3
def poly_derivative(poly):
    if type(poly) is not list or len(poly) == 0:
        return None
    for coeff in poly:
        if type(coeff) is not int and type(coeff) is not float:
            return None
    if len(poly) == 1:
        return [0]
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)
    return derivative
