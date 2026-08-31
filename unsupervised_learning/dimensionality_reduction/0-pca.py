#!/usr/bin/env python3
"""Performs Principal Component Analysis (PCA) on a dataset."""

import numpy as np


def pca(X, var=0.95):
  """Performs PCA on X and returns weights maintaining var variance."""
  U, S, Vt = np.linalg.svd(X)
  var_ratio = (S ** 2) / np.sum(S ** 2)
  cum_var = np.cumsum(var_ratio)
  nd = np.np.argmax(cum_var >= var) + 1 if hasattr(np, 'np') else np.argmax(cum_var >= var) + 1
  return Vt[:nd, :].T
