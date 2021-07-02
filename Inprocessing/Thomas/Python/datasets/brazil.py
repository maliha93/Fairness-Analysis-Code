import numpy as np
import pandas as pd
import os.path

from datasets.dataset import Dataset

BASE_URL = os.path.join('datasets', 'brazil', 'brazil.npz')

def load(gpa_cutoff=3.0, r_train=0.4, r_candidate=0.2, seed=None, include_intercept=True, use_pct=1.0, include_T=False, standardize=False):
	random = np.random.RandomState(seed)
	
	D = np.load(BASE_URL)
	X = D['X']
	Y = (D['Y']>=gpa_cutoff).astype(float) - (D['Y']<gpa_cutoff).astype(float)
	T = D['T']

	# Reduce the dataset size as needed
	n_keep = int(np.ceil(len(X) * use_pct))
	I = np.arange(len(X))
	random.shuffle(I)
	I = I[:n_keep]
	X = X[I]
	Y = Y[I].flatten()
	T = T[I].flatten()

	# Compute split sizes
	n_samples   = len(X)
	n_train     = int(r_train*n_samples)
	n_test      = n_samples - n_train
	n_candidate = int(r_candidate*n_train)
	n_safety    = n_train - n_candidate

	return Dataset(X, Y, T, n_candidate, n_safety, n_test, seed=seed, include_intercept=include_intercept, include_T=include_T, standardize=standardize)

