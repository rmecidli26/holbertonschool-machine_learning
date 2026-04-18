#!/usr/bin/env python3

def summation_i_squared(n):
    # n tam ədəd olmalıdır
    if not isinstance(n, int):
        return None
    
    # n mənfi ədəd ola bilməz (adətən tapşırıq şərtində belə olur)
    if n < 0:
        return None
    
    # n = 0 halı üçün
    if n == 0:
        return 0
    
    # n(n+1)(2n+1) / 6 düsturu
    # // istifadə edirik ki, nəticə float yox, tam olaraq int (məs: 14) olsun
    return (n * (n + 1) * (2 * n + 1)) // 6
