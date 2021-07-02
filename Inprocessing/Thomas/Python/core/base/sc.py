import numpy as np
from functools import partial

from core.optimizers import OPTIMIZERS
from utils.rvs import ConstraintManager

from sys import float_info
MAX_VALUE = float_info.max

class SeldonianClassifierBase:
	def __init__(self, constraint_strs, shape_error, model_type, model_params, verbose=False, ci_mode='hoeffding', term_values={}, cs_scale=2.0):
		self.shape_error = shape_error
		self.verbose     = verbose
		self.model_type      = model_type
		self.model_variables = {}
		self.model_params    = model_params
		self.ci_mode = ci_mode
		self._term_values = term_values
		self._cs_scale = cs_scale
		# Set up the constraint manager to handle the input constraints
		self._cm = ConstraintManager(['X', 'Y', 'Yp', 'T'], constraint_strs, trivial_bounds={}, keywords=self.keywords)

	@property
	def n_constraints(self):
		return self._cm.n_constraints
	

	# Error functions

	def _error(self, predictf, X, Y):
		return np.mean(Y != predictf(X))

	def _loss(self, X, Y, theta=None):
		predictf = self.get_predictf(theta=theta)
		return self._error(predictf, X, Y)


	# Accessor for different optimizers

	def get_optimizer(self, name, dataset, opt_params={}):
		if name == 'linear-shatter':
			assert self.model_type == 'linear', 'SeldonianClassifierBase.get_optimizer(): linear-shatter optimizer is only compatible with a linear model.'
			return OPTIMIZERS[name](dataset.X, buffer_angle=5.0, has_intercept=False, use_chull=True)
		elif name == 'cmaes':
			# return OPTIMIZERS[name](self.n_features, sigma0=2.0, n_restarts=50)
			return OPTIMIZERS[name](self.n_features, sigma0=0.0001, n_restarts=50) #, **opt_params)
		elif name == 'bfgs':
			return OPTIMIZERS[name](self.n_features, sigma0=2.0, n_restarts=50, **opt_params)
		elif name == 'slsqp':
			return OPTIMIZERS[name](self.n_features)
		raise ValueError('SeldonianClassifierBase.get_optimizer(): Unknown optimizer \'%s\'.' % name)

	
	# Base models	

	def get_predictf(self, theta=None):
		return partial(self.predict, theta=theta)

	def predict(self, X, theta=None):
		theta = self.theta if (theta is None) else theta
		if self.model_type == 'linear':
			return np.sign(X.dot(theta))
		elif self.model_type == 'rbf':
			s, b, *c = theta
			X_ref = self.model_variables['X']
			Y_ref = self.model_variables['Y']
			P = np.sum((X_ref[:,None,:]-X[None,:,:])**2, axis=2)
			K = np.exp(-0.5*P/(s**2))
			return np.sign(np.einsum('n,n,nm->m', c, Y_ref, K)+b)
		elif self.model_type == 'mlp':
			A = X
			for nh in self.model_params['hidden_layers']:
				nw = A.shape[0] * nh
				w, theta = theta[:nw], theta[nw:]
				W = w.reshape((A.shape[0], nh))
				A = np.hstack((A.dot(W),np.ones((W.shape[1],1))))
				A = np.tanh(A)
			return np.sign(A.dot(theta))
		raise ValueError('SeldonianClassifierBase.predict(): unknown model type: \'%s\''%self.model_type)

	@property
	def n_features(self):
		if self.model_type == 'linear':
			return self.dataset.n_features
		if self.model_type == 'rbf':
			return self.dataset.n_optimization + 2
		if self.model_type == 'mlp':
			n = 0
			n0 = self.dataset.n_features-1
			for nh in self.model_params['hidden_layers']:
				n += (n0+1) * nh
				n0 = nh
			n += nh+1
			return n

	def _store_model_variables(self, dataset):
		if self.model_type == 'rbf':
			X, Y, _ = dataset.optimization_splits()
			self.model_variables['X'] = X
			self.model_variables['Y'] = Y

	# Constrain evaluation

	def safety_test(self, predictf, Xs, Ys, Ts):
		try:
			Yp = predictf(Xs)
		except TypeError as e:
			return np.array([np.inf])
		data = {
			'X'  : Xs,
			'Y'  : Ys, 
			'Yp' : Yp,
			'T'  : Ts
		}
		# data = self.preprocessor.process(self._cm.base_variables, Xs, Ys, Yp, Ts)
		self._cm.set_data(data)
		return self._cm.upper_bound_constraints(self.deltas, mode=self.ci_mode, interval_scaling=1.0, term_values=self._term_values)
		
	def predict_safety_test(self, predictf, Xc, Yc, Tc, data_ratio):
		try:
			Yp = predictf(Xc)
		except TypeError as e:
			return np.array([np.inf])
		# data = self.preprocessor.process(self._cm.base_variables, Xc, Yc, Yp, Tc)
		data = {
			'X'  : Xc,
			'Y'  : Yc, 
			'Yp' : Yp,
			'T'  : Tc
		}
		self._cm.set_data(data)
		return self._cm.upper_bound_constraints(self.deltas, mode=self.ci_mode, interval_scaling=self._cs_scale, n_scale=data_ratio, term_values=self._term_values)

	# Candidate selection

	def candidate_objective(self, theta, X, Y, T, data_ratio):
		predictf = self.get_predictf(theta)
		sc_ubs   = self.predict_safety_test(predictf, X, Y, T, data_ratio)
		if any(np.isnan(sc_ubs)):
			return MAX_VALUE
		elif (sc_ubs <= 0.0).all():
			return self._loss(X, Y, theta=theta)
		elif self.shape_error:
			sc_ub_max = np.maximum(np.minimum(sc_ubs, MAX_VALUE), 0.0).sum()
			return 1.0 + sc_ub_max
		return 1.0

	# Model training

	def fit(self, dataset, n_iters=1000, optimizer_name='linear-shatter', theta0=None, opt_params={}):
		self.dataset = dataset

		# Get the optimizer
		opt = self.get_optimizer(optimizer_name, dataset, opt_params=opt_params)

		# Store any variables that will be required by the model
		self._store_model_variables(dataset)
		
		# Compute number of samples of g(theta) for the candidate and safety sets
		# Note: This assumes that the number of samples used to estimate g(theta)
		#       doesn't depend on theta itself.  
		data_ratio = dataset.n_safety/dataset.n_optimization

		# Fix the non-theta arguments of the candidate objective
		Xc, Yc, Tc = dataset.optimization_splits()
		c_objective  = partial(self.candidate_objective, X=Xc, Y=Yc, T=Tc, data_ratio=data_ratio)

		# Perform candidate selection using the optimizer
		self.theta,_ = opt.minimize(c_objective, n_iters, theta0=theta0)
		
		# Perform the safety check to determine if theta can be used
		predictf = self.get_predictf()
		st_thresholds = self.safety_test(predictf, *dataset.safety_splits())
		accept = False if any(np.isnan(st_thresholds)) else (st_thresholds <= 0.0).all()
		return accept

	# Model evaluation

	def evaluate(self, dataset, predictf=None, override_is_seldonian=False):
		ds_ratio = dataset.n_safety / dataset.n_optimization
		meta   = {}
		splits = {'candidate' : dataset.optimization_splits(),
				  'safety'    : dataset.safety_splits(),
				  'train'     : dataset.training_splits(),
				  'test'      : dataset.testing_splits()}

		# We don't assume to know what model predictf uses, so assume that
		#   that any predictf passed in as an argument is Non-Seldonian
		if predictf is None:
			predictf = self.get_predictf()
			meta['is_seldonian'] = True
		else:
			meta['is_seldonian'] = override_is_seldonian

		# Record statistics for each split of the dataset
		for name, (X,Y,T) in splits.items():
			try:
				Yp = predictf(X)
			except TypeError as e:
				meta['loss_%s' % name] = np.nan
				for cnum in range(self.n_constraints):
					meta['co_%d_mean' % cnum] = np.nan
			else:
				meta['loss_%s' % name] = self._error(predictf, X, Y)			
				# data = self.preprocessor.process(self._cm.base_variables, X, Y, Yp, T)
				data = {
					'X'  : X,
					'Y'  : Y, 
					'Yp' : Yp,
					'T'  : T
				}
				self._cm.set_data(data)
				values = self._cm.evaluate()
				for cnum, value in enumerate(values):
					meta['co_%d_mean' % cnum] = value
		
		# Record SMLA-specific values or add baseline defaults
		if meta['is_seldonian']:
			stest      = self.safety_test(predictf, *splits['safety'])
			pred_stest = self.predict_safety_test(predictf, *splits['candidate'], ds_ratio)
			meta['accept']                = False if any(np.isnan(stest))      else (     stest <= 0.0).all()
			meta['predicted_accept']      = False if any(np.isnan(pred_stest)) else (pred_stest <= 0.0).all()
			for i,(st, pst) in enumerate(zip(stest, pred_stest)):
				meta['co_%d_safety_thresh'  % i] = st
				meta['co_%d_psafety_thresh' % i] = pst
		else:
			meta['accept']                = True
			meta['predicted_accept']      = True
			for i in range(self.n_constraints):
				meta['co_%d_safety_thresh'  % i] = np.nan
				meta['co_%d_psafety_thresh' % i] = np.nan
		return meta

########################
#   Base Classifiers   #
########################

class SeldonianClassifier(SeldonianClassifierBase):
	def __init__(self, constraint_strs, deltas, shape_error=False, verbose=False, model_type='linear', model_params={}, ci_mode='hoeffding', n_types=2, term_values={}, cs_scale=2.0):
		self.deltas = deltas
		self.keywords = {
			'FPR' : 'E[Yp=1|Y=-1]',
			'FNR' : 'E[Yp=-1|Y=1]',
			'TPR' : 'E[Yp=1|Y=1]',
			'TNR' : 'E[Yp=-1|Y=-1]',
			'PR'  : 'E[Yp=1]',
			'NR'  : 'E[Yp=-1]'
		}

		super().__init__(constraint_strs, shape_error=shape_error, verbose=verbose, model_type=model_type, model_params=model_params, ci_mode=ci_mode, term_values=term_values, cs_scale=cs_scale)
		
class SeldonianMCClassifier(SeldonianClassifierBase):
	def __init__(self, epsilons, deltas, shape_error=False, verbose=False, model_type='linear', n_classes=2, model_params={}, loss_weights=None, term_values={}):
		self.n_classes = n_classes
		self.loss_weights = 1 - np.eye(n_classes) if (loss_weights is None) else loss_weights
		self.constraint_weights = np.zeros_like(self.loss_weights)
		self.constraint_weights[0,1] = 1.0
		self.epsilons = epsilons
		self.deltas   = deltas
		super().__init__(shape_error=shape_error, verbose=verbose, model_type=model_type, model_params=model_params, term_values=term_values)

	def predict(self, X, theta=None):
		theta = self.theta if (theta is None) else theta
		if self.model_type == 'linear':
			theta = theta.reshape((X.shape[1],self.n_classes))
			return np.argmax(X.dot(theta), axis=1)
		raise ValueError('GeneralSeldonianMCClassifier.predict(): unknown model type: \'%s\''%self.model_type)

	def _get_confusion_indicators(self, Y, Yp):
		n = len(Y)
		C  = np.zeros((n,self.n_classes,self.n_classes))
		C[np.arange(n), Yp, Y] += 1
		return C

	def _error(self, predictf, X, Y):
		Yp = predictf(X)
		C = self._get_confusion_indicators(Y, Yp)
		return (C.sum(0) * self.loss_weights).sum() / C.shape[0]

	@property
	def n_features(self):
		if self.model_type == 'linear':
			return self.dataset.n_features * self.n_classes


