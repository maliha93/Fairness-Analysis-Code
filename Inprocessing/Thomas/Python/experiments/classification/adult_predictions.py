import numpy as np
import warnings
import argparse
import os
import time
import pandas as pd

from scipy.optimize import OptimizeWarning
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning

# SeldonianML imports
from utils.rvs import ConstraintManager
from datasets  import brazil as brazil
from datasets  import propublica as compas
from datasets  import adult as adult
from core.base.sc   import SeldonianClassifier

# fair classification imports for pretraining
import baselines.fair_classification.utils as fc_ut
import baselines.fair_classification.loss_funcs as fc_lf

# Supress sklearn FutureWarnings for SGD
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=OptimizeWarning)
warnings.simplefilter(action='ignore', category=LinAlgWarning)


def pretrain(dataset, args):
	print('   Pretraining...')
	apply_fairness_constraints = 1
	apply_accuracy_constraint  = 0
	sep_constraint = 0
	gamma = None
	e = -args.e*100
	X, Y, T = dataset.training_splits()
	sensitive_attrs_to_cov_thresh = {'T':0.01}
	w = fc_ut.train_model(X, Y, {'T':T.astype(np.int64)}, fc_lf._logistic_loss, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, ['T'], sensitive_attrs_to_cov_thresh, gamma)
	return w

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	#parser.add_argument('--gpa_cutoff',     type=float, default=3.5,               help='Cutoff for defining "good" GPA.')
	parser.add_argument('--definition',     type=str,   default='DisparateImpact', help='Choice of safety definition to enforce.')
	parser.add_argument('--e',              type=float, default=0.05,              help='Value for epsilon.')
	parser.add_argument('--d',              type=float, default=0.05,              help='Value for delta.')
	parser.add_argument('--data_pct',       type=float, default=1.0,               help='Proportion of the overall size of the dataset to use.')
	parser.add_argument('--r_train_v_test', type=float, default=0.4,               help='Ratio of data used for training vs testing.')
	parser.add_argument('--r_cand_v_safe',  type=float, default=0.4,               help='Ratio of training data used for candidate selection vs safety checking.')
	parser.add_argument('--pretrain',       action='store_true',                   help='Whether or not to pretrain using fair classification to accelerate candidate selection.')
	parser.add_argument('--n_iters',        type=int,   default=5000,              help='Number of SMLA training iterations.')
	args = parser.parse_args()
	args_dict = dict(args.__dict__)
	
	# Load the dataset based on input parameters
	dataset = adult.load(use_pct=args.data_pct, r_train=args.r_train_v_test, r_candidate=args.r_cand_v_safe, include_intercept=True)

	# Set up the definition of fairness to use based on args.definition and args.e
	all_constraint_strs = {
		'DemographicParity'  : (lambda e: "|PR(T=0)-PR(T=1)| - %f" % e),
		'DisparateImpact'    : (lambda e: "-min(PR(T=0)/PR(T=1), PR(T=1)/PR(T=0)) - %f" % e),
		'EqualizedOdds'      : (lambda e: "|TPR(T=0)-TPR(T=1)| + |FPR(T=0)-FPR(T=1)| - %f" % e),
		'EqualOpportunity'   : (lambda e: "|FNR(T=0)-FNR(T=1)| - %f" % e),
		'PredictiveEquality' : (lambda e: "|FPR(T=0)-FPR(T=1)| - %f" % e),
	}
	constraint_str = all_constraint_strs[args.definition](args.e)

	print('Generating predictions for constraint:  \'%s\'' % constraint_str)
	s = time.time()

	# Pretrain using fair classification if desired (Useul for accelerating SeldonianML's candidate selection step)
	#w = pretrain(dataset,args) if args.pretrain else None
	w = None

	# Train SC using hoeffding's inequality
	print('   Training Hoeffding SC...')
	model_h = SeldonianClassifier([constraint_str], [args.d], ci_mode='hoeffding', shape_error=True, model_type='linear')
	model_h.fit(dataset, n_iters=args.n_iters, optimizer_name='cmaes', theta0=w)

	# Train SC using inversion of the student's t-Test
	print('   Training t-Test SC...')
	model_t = SeldonianClassifier([constraint_str], [args.d], ci_mode='ttest', shape_error=True, model_type='linear')
	model_t.fit(dataset, n_iters=args.n_iters, optimizer_name='cmaes', theta0=w)
	
	# Gather predictions on the test set
	print('   Collecting predictions...')
	X, Y, T = dataset.testing_splits()
	Yp_h = model_h.predict(X)
	Yp_t = model_t.predict(X)
	e = time.time()
	print("Time:", (e-s))
	df = pd.read_csv("datasets/adult/Adult.csv")
	df = df.iloc[df.shape[0]-X.shape[0]:, :]
	df['pred_h'] = Yp_h
	df['pred_t'] = Yp_t
	df.to_csv("results_Thomas/adult_test_"+args.definition.lower()+"_repaired.csv", index=False)   

	# Collect output and save results 
	output = {
		'X' : X,
		'Y' : Y,
		'T' : T,
		'Yp_h' : Yp_h,
		'Yp_t' : Yp_t
	}
	#fname = 'results/adult_predictions/%s.npz' % args.definition.lower()
	#os.makedirs('results/adult_predictions/', exist_ok=True)
	#np.savez(fname, **output)
	print('   Done. Results saved.')
