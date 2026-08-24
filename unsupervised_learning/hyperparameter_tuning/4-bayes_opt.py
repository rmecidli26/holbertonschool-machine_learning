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
        
        # Standard deviation qrafikində mənfi və ya sıfır qiymətlərin qarşısını almaq
        sigma = np.maximum(sigma, 1e-9)
        sigma_sig = np.sqrt(sigma)

        if self.minimize:
            current_optimum = np.min(self.gp.Y)
            improvement = current_optimum - mu - self.xsi
        else:
            current_optimum = np.max(self.gp.Y)
            improvement = mu - current_optimum - self.xsi

        Z = improvement / sigma_sig
        EI = improvement * norm.cdf(Z) + sigma_sig * norm.pdf(Z)

        # Standart kənarlaşma sıfır olduqda EI-ni sıfırlamaq
        EI[sigma_sig == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
