def summation_i_squared(n):
    if not isinstance(n, (int, float)) or n < 0:
        return None
    
    n = int(n)
    # Formül: n * (n + 1) * (2n + 1) / 6
    result = (n * (n + 1) * (2 * n + 1)) // 6
    return result
