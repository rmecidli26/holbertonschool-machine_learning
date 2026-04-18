def summation_i_squared(n):
    if not n.isdigit():
        return None
    sum = 0
    for i in range(0, n):
        sum = sum + i*i
    return sum
