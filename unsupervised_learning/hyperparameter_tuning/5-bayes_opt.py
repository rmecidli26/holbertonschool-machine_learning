#!/usr/bin/env python3
"""Bayesian Optimization implementation using a Gaussian Process.
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Represents a Bayesian optimization on a 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize the Bayesian Optimization."""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculate the next best sample location
        using Expected Improvement (EI).
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            current_optimum = np.min(self.gp.Y)
            improvement = current_optimum - mu - self.xsi
        else:
            current_optimum = np.max(self.gp.Y)
            improvement = mu - current_optimum - self.xsi

        Z = np.zeros_like(mu)
        EI = np.zeros_like(mu)

        # Sıfıra bölünmənin və ya riyazi xətaların qarşısını almaq üçün maska
        mask = sigma > 0

        # Z dəyərinin sadəcə etibarlı dispersiya olduqda hesablanması
        Z[mask] = improvement[mask] / sigma[mask]

        # W503/W504 xətalarından qaçmaq üçün riyazi ifadə iki hissəyə bölünür
        term1 = improvement[mask] * norm.cdf(Z[mask])
        term2 = sigma[mask] * norm.pdf(Z[mask])
        EI[mask] = term1 + term2

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function."""
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Təklif olunan növbəti nöqtə artıq mövcuddursa, optimallaşdırmanı
            # dayandırırıq (erkən dayanma).
            if X_next in self.gp.X:
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        return self.gp.X[idx_opt], self.gp.Y[idx_opt]
