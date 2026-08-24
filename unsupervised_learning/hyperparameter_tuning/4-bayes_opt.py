#!/usr/bin/env python3
"""Bayesian Optimization acquisition implementation using a Gaussian Process.
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Represents a Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize the Bayesian Optimization."""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculate the next best sample location using Expected Improvement."""
        mu, sigma = self.gp.predict(self.X_s)
        sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero
        sigma = np.sqrt(sigma)

        if self.minimize:
            current_optimum = np.min(self.gp.Y)
            improvement = current_optimum - mu - self.xsi
        else:
            current_optimum = np.max(self.gp.Y)
            improvement = mu - current_optimum - self.xsi

        Z = improvement / sigma
        EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        # If sigma is 0, EI should be 0
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
