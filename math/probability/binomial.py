#!/usr/bin/env python3
"""
Binomial distribution module
"""


class Binomial:
    """ Represents a binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ Initialize Binomial distribution """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and variance of data
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Estimate p and n: mean = n*p, variance = n*p*(1-p)
            # p = 1 - (variance / mean)
            estimated_p = 1 - (variance / mean)
            self.n = int(round(mean / estimated_p))
            # Recalculate p based on rounded n
            self.p = float(mean / self.n)
