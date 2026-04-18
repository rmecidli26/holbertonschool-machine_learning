#!/usr/bin/env python3
"""Module to calculate the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    if type(C) is not int:
        return None

    # İnteqralın başlanğıcı inteqral sabiti C ilə başlayır
    integral = [C]

    for i in range(len(poly)):
        if type(poly[i]) is not int and type(poly[i]) is not float:
            return None
        
        # Yeni əmsal: a_i / (i + 1)
        new_coeff = poly[i] / (i + 1)
        
        # Əgər əmsal tam ədəddirsə, onu integer kimi saxla
        if new_coeff.is_integer():
            new_coeff = int(new_coeff)
            
        integral.append(new_coeff)

    # Siyahının sonunda artıq 0-lar varsa onları təmizləyirik (as small as possible)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
