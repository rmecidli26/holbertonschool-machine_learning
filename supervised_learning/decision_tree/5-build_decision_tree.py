#!/usr/bin/env python3
import numpy as np

# Class daxilində metod tam olaraq belə olmalıdır:
    def update_indicator(self):
        def is_large_enough(x):
            if not self.lower: return np.ones(x.shape[0], dtype=bool)
            return np.all([x[:, k] > v for k, v in self.lower.items()], axis=0)

        def is_small_enough(x):
            if not self.upper: return np.ones(x.shape[0], dtype=bool)
            return np.all([x[:, k] <= v for k, v in self.upper.items()], axis=0)

        self.indicator = lambda x : np.all([is_large_enough(x), is_small_enough(x)], axis=0)
