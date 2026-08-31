#!/usr/bin/env python3
import numpy as np


def pca(X, var=0.95):
  """Performs PCA on a dataset keeping 'var' fraction of variance."""
  # Compute SVD of X
  U, S, Vt = np.linalg.svd(X)

  # Compute explained variance ratio
  explained_variance = (S ** 2) / np.sum(S ** 2)
  cumulative_variance = np.cumsum(explained_variance)

  # Find number of components to reach 'var'
  nd = np.argmax(cumulative_variance >= var) + 1

  # Return the weights matrix W (transpose of first nd rows of Vt)
  W = Vt[:nd, :].T
  return W
