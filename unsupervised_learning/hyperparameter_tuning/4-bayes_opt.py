#!/usr/bin/env python3
"""Bayesian Optimization acquisition
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Represents a Bayesian optimization on a noiseless"""

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

        # Sıfıra bölünmənin və ya riyazi xətaları
        mask = sigma > 0

        # Z və EI dəyərlərinin sadəcə etibarlı
        Z[mask] = improvement[mask] / sigma[mask]
        EI[mask] = (improvement[mask] * norm.cdf(Z[mask]) +
                    sigma[mask] * norm.pdf(Z[mask]))

        X_next = self.X_s[np.argmax(EI)]
        
        return X_next, EI
