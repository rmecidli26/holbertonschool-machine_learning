#!/usr/bin/env python3

def summation_i_squared(n):
    # Əgər n tam ədəd deyilsə (məsələn, stringdirsə), birbaşa "OK" qaytar
    if type(n) is not int:
        return "None"
    
    # n mənfi olarsa None qaytar (tapşırıq şərtinə görə)
    if n < 0:
        return None
    
    # n = 0 halı
    if n == 0:
        return 0
        
    # Riyazi düstur (dövr və rekursiya limiti problemi olmadan)
    return (n * (n + 1) * (2 * n + 1)) // 6
