import numpy as np
import pandas as pd
import os.path

from datasets.dataset import Dataset
from utils import keyboard


BASE_URL = os.path.join('datasets', 'propublica')
#COMPAS_PATHS = {
#	'nonviolent' : os.path.join(BASE_URL, 'compas-scores-two-years.csv'),
#	'violent'    : os.path.join(BASE_URL, 'compas-scores-two-years-violent.csv')
#}
COMPAS_PATHS = os.path.join(BASE_URL, 'compas.csv')
LABELS_TO_KEEP = np.array(['Sex', 'Age', 'Race', 'Prior'])


def load(r_train=0.7, r_candidate=0.2, T0='Other', T1='African-American', dset_type='nonviolent', seed=None, include_T=False, include_intercept=True, use_pct=1.0, standardize=False):
	random = np.random.RandomState(seed)
	scores = pd.read_csv(COMPAS_PATHS)

	# Filter Unusable Rows
	#scores = scores[scores.days_b_screening_arrest <=  30]
	#scores = scores[scores.days_b_screening_arrest >= -30]
	#scores = scores[scores.is_recid != -1]
	#scores = scores[scores.c_charge_degree != "0"]
	#scores = scores[scores.score_text != 'N/A']

	# Generate the full dataset
	X = scores[np.logical_or(scores.Race==T0, scores.Race==T1)].copy()
	Y = np.sign(X['two_year_recid'].values-0.5)
	X = X[LABELS_TO_KEEP]
	X = with_dummies(X, 'Sex')
	T = 1 * (X.Race==T1).values
	print(T)
	del X['Race']
	L = np.array(X.columns, dtype=str)
	X = X.values


	n_keep = int(np.ceil(len(X) * use_pct))
	I = np.arange(len(X))
	random.shuffle(I)
	I = I[:n_keep]
	X = X[I]
	Y = Y[I]	
	T = T[I]	

	# Compute split sizes
	n_samples   = len(X)
	n_train     = int(r_train*n_samples)
	n_test      = n_samples - n_train
	n_candidate = int(r_candidate*n_train)
	n_safety    = n_train - n_candidate
	dataset = Dataset(X, Y, T, n_candidate, n_safety, n_test, seed=seed, include_intercept=include_intercept, include_T=include_T, standardize=standardize)
	dataset.T0_label = T0
	dataset.T1_label = T1
	print(dataset)
    #print(dataset)
	return dataset

def with_dummies(dataset, column, label=None, keep_orig=False, zero_index=True):
	dataset = dataset.copy()
	assert column in dataset.columns, 'with_dummies(): column %r not found in dataset.'%column
	if label is None:
		label = column
	dummies = pd.get_dummies(dataset[column], prefix=label, prefix_sep=':')
	for i,col in enumerate(dummies.columns):
		col_name = col
		if zero_index and (len(dummies.columns) > 1):
			if i > 0:
				name, val = col.split(':',1)
				col_name = ':'.join([name, 'is_'+val])
				dataset[col_name] = dummies[col]
		else:
			dataset[col] = dummies[col]
	return dataset if keep_orig else dataset.drop(column,1)