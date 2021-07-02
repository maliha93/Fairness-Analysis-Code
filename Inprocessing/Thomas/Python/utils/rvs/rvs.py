import numpy as np
import itertools 
from scipy.stats   import norm, chi, t
from scipy.special import erf, erfinv
from scipy.stats import beta

from time import time
from copy import copy, deepcopy

from parglare import Parser, Grammar

import re
import warnings


COMPARATORS = {
	'>'  : lambda a, b: a > b,
	'<'  : lambda a, b: a < b,
	'>=' : lambda a, b: a >= b,
	'<=' : lambda a, b: a >= b,
	'='  : lambda a, b: a == b,
	'!=' : lambda a, b: np.logical_not(a==b)
}

def get_constant_name(counter={'c':0}):
	name = 'c%d' % counter['c']
	counter['c'] += 1
	return name

def get_variable_name(counter={'v':0}):
	name = 'v%d' % counter['v']
	counter['v'] += 1
	return name

def get_expression_name(counter={'e':0}):
	name = 'e%d' % counter['c']
	counter['e'] += 1
	return name


def parse_value(value):
	''' Attempts to interpret <value> as a number. '''
	if isinstance(value, str):
		try:
			value = int(value)
		except ValueError:
			value = float(value)
	return value

class RVFuncs():
	@staticmethod
	def constant(value_raw):
		value = parse_value(value_raw)
		if value >= 0:
			return ConstantExpression('c', value)
		return RVFuncs.negative(ConstantExpression('c', -value))

	@staticmethod
	def variable(name):
		return VariableExpression(name)

	@staticmethod
	def comparator_variable(term1, comp, term2):
		return ComparatorExpression(term1, comp, term2)

	@staticmethod
	def sample_set(variable, condition=None):
		return SampleSet(variable, condition)

	@staticmethod
	def expected_value(sampleset):
		return ExpectedValue(sampleset)

	@staticmethod
	def negative(e):
		''' Returns the negative of <e>, reducing nested negatives. '''
		n_negatives = 1
		while isinstance(e, NegativeExpression):
			e = e._terms[0]
			n_negatives += 1
		return NegativeExpression(e) if (n_negatives % 2 == 1) else e

	@staticmethod
	def sum(*expressions):
		''' Returns the sum of <expressions>, factoring out constants and shared factors. '''
		# Aggregate terms that are sums themselves
		exps = []
		for e in expressions:
			if isinstance(e, SumExpression):
				exps.extend(e._terms)
			else:
				exps.append(e)
		expressions = exps
				
		# Aggregate terms that are constants
		cval = 0
		exps = []
		for e in expressions:
			if isinstance(e, ConstantExpression):
				cval += e.value
			elif isinstance(e, NegativeExpression) and isinstance(e._terms[0], ConstantExpression):
				cval -= e._terms[0].value
			else:
				exps.append(e)
		if cval != 0 or len(exps) == 0:
			const = RVFuncs.constant(cval)
			exps = [ const, *exps]
		expressions = exps

		if len(expressions) == 1:
			return expressions[0]




		# Check if all terms share a common denominator and factor it out
		def split_as_fraction(e):
			if isinstance(e, FractionExpression):
				return [e._terms[0], e._terms[1]]
			elif isinstance(e, NegativeExpression) and isinstance(e._terms[0],FractionExpression):
				return [RVFuncs.negative(e._terms[0]._terms[0]), e._terms[0]._terms[1]]
			return [e, None]
		nums, dens = zip(*[ split_as_fraction(e) for e in exps ])
		if all([ not(dens[0] is None) and d==dens[0] for d in dens ]):
			exps = nums
			common_den = dens[0]
		else:
			common_den = None

		# Check if any terms have shared product factors and factor them out
		def extract_unsigned_terms(e):
			if isinstance(e, NegativeExpression) or isinstance(e, FractionExpression):
				return extract_unsigned_terms(e._terms[0])
			if isinstance(e, ProductExpression):
				return e._terms
			return [e]
		def remove_terms(e, terms):
			if isinstance(e, NegativeExpression):
				return RVFuncs.negative(remove_terms(e._terms[0], terms))
			if isinstance(e, FractionExpression):
				return RVFuncs.fraction(remove_terms(e._terms[0], terms), e._terms[1])
			if isinstance(e, ProductExpression):
				remaining = e._terms.copy()
				for t in terms:
					remaining.remove(t)
				return RVFuncs.product(*remaining) if len(remaining) > 0 else RVFuncs.constant(1)
			return RVFuncs.constant(1) if len(terms) > 0 else e
		has_negative   = [ isinstance(e,NegativeExpression) for e in exps ]
		unsigned_terms = [ extract_unsigned_terms(e)        for e in exps ]
		unsigned_terms_tmp = deepcopy(unsigned_terms)
		shared_terms = []
		for st in unsigned_terms[0]:
			if isinstance(st, ConstantExpression) and (st.value == 1):
				continue
			if all([ (st in terms) for terms in unsigned_terms_tmp[1:] ]):
				shared_terms.append(st)
				for terms in unsigned_terms_tmp:
					terms.remove(st)
		if len(shared_terms) > 0:
			remainder = RVFuncs.sum(*[ remove_terms(e, shared_terms) for e in exps ])
		else:
			remainder = SumExpression(exps)

		# Return the product of the common factor and the remainder sum
		if len(shared_terms) > 0 and common_den is None:
			common_factor = RVFuncs.product(*shared_terms)
			return RVFuncs.product(common_factor, remainder)
		elif len(shared_terms) > 0:
			common_factor = RVFuncs.fraction(RVFuncs.product(*shared_terms), common_den)
			return RVFuncs.product(common_factor, remainder)
		return remainder

	@staticmethod
	def diff(e0, e1):
		return RVFuncs.sum(e0, RVFuncs.negative(e1))

	@staticmethod
	def max(*expressions):
		if len(expressions) == 1:
			return expressions[0]
		exps = []
		for e in expressions:
			if isinstance(e, MaxExpression):
				exps.extend(e._terms)
			else:
				exps.append(e)
		if len(expressions) == 2:
			e1, e2 = expressions
			# If the max *happens* to be Max(E, 1/E) for some E, reduce to a MaxRecip
			if e1 == RVFuncs.fraction(RVFuncs.constant(1), e2):
				return MaxRecipExpression(e1)
			# If the max *happens* to be Max(E, -E) for some E, reduce to Abs
			elif e1 == RVFuncs.negative(e2):
				return AbsExpression(e1)
		return MaxExpression(exps)

	@staticmethod
	def min(*expressions):
		if len(expressions) == 1:
			return expressions[0]
		exps = []
		for e in expressions:
			if isinstance(e, MaxExpression):
				exps.extend(e._terms)
			else:
				exps.append(e)
		# Convert to a negative max
		exps = [ RVFuncs.negative(e) for e in exps ]
		return RVFuncs.negative(RVFuncs.max(*exps))

	@staticmethod
	def abs(e):
		if isinstance(e, NegativeExpression):
			e = e._terms[0]
		return AbsExpression(e)

	@staticmethod
	def pow(e, c):
		return e  # fix

	@staticmethod
	def logical_and(comparisons):
		return AndExpression(comparisons)

	@staticmethod
	def product(*expressions):

		# Strip negatives from input expressions
		n_negatives = 0
		exps = []
		for e in expressions:
			if isinstance(e, NegativeExpression):
				exps.append(e._terms[0])
				n_negatives += 1
			else:
				exps.append(e)
		expressions = exps

		# Remove and input expressions that are a constant 1
		exps = []
		for e in expressions:
			if not(isinstance(e, ConstantExpression) and (e.value == 1)):
				exps.append(e)
		expressions = exps

		# # If there is only one input expression remaining, just return it
		# if len(expressions) == 1:
		# 	return RVFuncs.negative(expressions[0]) if n_negatives % 2 == 1 else expressions[0]

		# If any of the input expressions are a constant equal to 0, return 0
		if any([ isinstance(e,ConstantExpression) and (e.value==0) for e in expressions ]):
			return RVFuncs.constant(0)

		# Aggregate input expressions that are products or fractions
		num_exps = []
		den_exps = []
		for e in expressions:
			if isinstance(e, ProductExpression):
				num_exps.extend(e._terms)
			elif isinstance(e, FractionExpression):
				num_exps.append(e._terms[0])
				den_exps.append(e._terms[1])
			else:
				num_exps.append(e)

		if len(den_exps) > 0:
			# We have a fraction
			num = RVFuncs.product(*num_exps) if len(num_exps) > 1 else num_exps[0]
			den = RVFuncs.product(*den_exps) if len(den_exps) > 1 else den_exps[0]
			expr = RVFuncs.fraction(num, den)
		else:
			# We have a non-fraction product
			# Aggregate constants
			cval = 1
			_exps = []
			for e in num_exps:
				if isinstance(e, ConstantExpression):
					cval = safeprod(cval, e.value)
				else:
					_exps.append(e)
			if len(_exps) == 0:
				expr = RVFuncs.constant(cval)
			elif cval != 1:
				_exps.append(RVFuncs.constant(cval))
				expr = ProductExpression(_exps)
			elif len(_exps) > 1:
				expr = ProductExpression(_exps)
			else:
				expr = _exps[0]

		return expr if (n_negatives % 2 == 0) else RVFuncs.negative(expr)

	@staticmethod
	def fraction(num, den):
		''' Process the numerator and denominator to produce a reduced expression of one of the following forms, in this priority:
		 	 Constant or Variable
			 Negative(Product(PositiveConstant, Fraction)) 
			 Product(PositiveConstant, Fraction)
			 Negative(Fraction).
			Assumes that num and den are already processed into Negative(Product(Constant, Expression)) form. '''

		# Simplify negative signs in the numerator/denominator
		n_negatives = 0
		if isinstance(num, NegativeExpression):
			num = num._terms[0]
			n_negatives += 1
		if isinstance(den, NegativeExpression):
			den = den._terms[0]
			n_negatives += 1

		# Remove any constants in front of the numerator or denominator
		num_val = 1
		den_val = 1
		if isinstance(num, ProductExpression) and isinstance(num._terms[0], ConstantExpression):
			num_val = num._terms[0].value
			num = RVFuncs.product(*num._terms[1:]) if len(num._terms) > 1 else RVFuncs.constant(1)
		if isinstance(den, ProductExpression) and isinstance(den._terms[0], ConstantExpression):
			den_val = den._terms[0].value
			den = RVFuncs.product(*den._terms[1:]) if len(den._terms) > 1 else RVFuncs.constant(1)
		cval = safediv(num_val, den_val)
		if cval < 0:
			n_negatives += 1
			cval = -cval

		# Aggregate terms in the numerator/denominator if one or both are already a fraction
		if isinstance(num, FractionExpression) and isinstance(den, FractionExpression):
			_num = RVFuncs.product(num._terms[0], den._terms[1])
			_den = RVFuncs.product(num._terms[1], den._terms[0])
			num, den = _num, _den
		elif isinstance(num, FractionExpression):	
			_num = num._terms[0]
			_den = RVFuncs.product(num._terms[1], den)
			num, den = _num, _den
		elif isinstance(den, FractionExpression):
			_num = RVFuncs.product(den._terms[1], num)
			_den = den._terms[0]
			num, den = _num, _den

		# Remove terms in products that are present in both the numerator and denominator
		expr = None
		if num == den:
			expr = RVFuncs.constant(1)
		elif isinstance(den, ConstantExpression) and den.value == 1:
			expr = num
		elif isinstance(num, ProductExpression) and isinstance(den, ProductExpression):
			nterms, dterms = copy(num._terms), copy(den._terms)
			for term in nterms:
				if term in den._terms:
					num._terms.remove(term)
					den._terms.remove(term)
			num = RVFuncs.constant(1) if len(num._terms) == 0 else RVFuncs.product(*num._terms)
			den = RVFuncs.constant(1) if len(den._terms) == 0 else RVFuncs.product(*den._terms)
			if isinstance(num, ConstantExpression) and isinstance(den, ConstantExpression):
				expr = RVFuncs.constant(safediv(num.value, den.value))
		elif isinstance(num, ProductExpression) and isinstance(den, SingleTermExpression): 
			if den in num._terms:
				num._terms.remove(den)
				expr = RVFuncs.product(*num._terms)
		elif isinstance(den, ProductExpression) and isinstance(num, SingleTermExpression): 
			if num in den._terms:
				den._terms.remove(num)
				den = RVFuncs.product(*den._terms)
				if isinstance(den, ConstantExpression):
					print(safediv(1,den.value), RVFuncs.constant(safediv(1,den.value)).value)
					expr = RVFuncs.constant(safediv(1,den.value))
				else:
					expr = FractionExpression(RVFuncs.constant(1), RVFuncs.product(*den._terms))
		if expr is None:
			expr = FractionExpression(num, den)


		# Add a constant scaling factor if it is not 1
		if cval != 1:
			constant = RVFuncs.constant(cval)
			expr = RVFuncs.product(constant, expr)
		return RVFuncs.negative(expr) if n_negatives % 2 == 1 else expr





class Expression():
	def __init__(self):
		self.trivial_bounds = None
		self._terms = []
	def __eq__(self, E):
		return isinstance(E, self.__class__) and all([ T==_T for (T,_T) in zip(self._terms,E._terms)])

class CommutativeExpression(Expression):
	def __init__(self):
		super().__init__()
	def __eq__(self,E):
		if not(isinstance(E, self.__class__)):
			return False
		terms, _terms = copy(self._terms), copy(E._terms)
		try:
			for term in terms:
				_terms.remove(term)
		except ValueError:
			return False
		return len(_terms) == 0

class NoncommutativeExpression(Expression):
	def __init__(self):
		super().__init__()
	def __eq__(self,E):
		return isinstance(E, self.__class__) and all([ T==_T for (T,_T) in zip(self._terms,E._terms) ])

class SingleTermExpression():
	pass


class SampleSet(Expression):
	def __init__(self, expression, condition=None):
		super().__init__()
		self.expression = expression
		self.condition  = condition


class ConstantExpression(Expression, SingleTermExpression):
	def __init__(self, name, value):
		super().__init__()
		self.name = get_constant_name()
		self.value = value
	def __repr__(self):
		return  str(self.value)
	def __eq__(self, E):
		return isinstance(E,self.__class__) and self.value == E.value

class VariableExpression(Expression, SingleTermExpression):
	def __init__(self, name):
		super().__init__()
		self.name = name
	def __repr__(self):
		return self.name
	def __eq__(self, E):
		return isinstance(E,self.__class__) and self.name == E.name

class SampleSet(Expression, SingleTermExpression):
	def __init__(self, expression, condition=None):
		super().__init__()
		name = '%r' % expression
		if not(condition is None):
			name += '|%r' % condition
		self.name = '[%s]' % name
		self.expression = expression
		self.condition  = condition
	def __repr__(self):
		return self.name
	def __eq__(self, E):
		return isinstance(E,self.__class__) and (self.expression == E.expression) and (self.condition == E.condition)


class ExpectedValue(Expression, SingleTermExpression):
	def __init__(self, sample_set):
		super().__init__()
		self.name = 'E' + sample_set.name
		self.sample_set = sample_set
	def __repr__(self):
		return self.name
	def __eq__(self, E):
		return isinstance(E,self.__class__) and self.sample_set == E.sample_set



class ComparatorExpression(VariableExpression):
	def __init__(self, term1, comp, term2):
		name = '%r %s %r' % (term1, comp, term2)
		super().__init__(name)
		self.variable = term1
		self.comparator = comp
		self.value = term2


class NegativeExpression(NoncommutativeExpression, SingleTermExpression):
	def __init__(self, expression):
		super().__init__()
		self._terms = [expression]
	def __repr__(self):
		if isinstance(self._terms[0], SumExpression):
			return '-(%r)' % self._terms[0]	
		return '-%r' % self._terms[0]
	def __eq__(self, E):
		if isinstance(E,self.__class__) and (self._terms[0]==E._terms[0]):
			return True
		if isinstance(E, SumExpression):
			return E == self
		return False


class AbsExpression(NoncommutativeExpression, SingleTermExpression):
	def __init__(self, expression):
		super().__init__()
		self._terms = [expression]
	def __repr__(self):
		return '|%r|' % self._terms[0]	
	def __eq__(self, E):
		return isinstance(E,self.__class__) and (self._terms[0]==E._terms[0]) 

class FractionExpression(NoncommutativeExpression):
	def __init__(self, num, den):
		super().__init__()
		self._terms = [num, den]
	def __repr__(self):
		num, den = self._terms
		num_str = '(%r)'%num if isinstance(num, SumExpression) else '%r'%num
		den_str = '%r'%den if isinstance(den, SingleTermExpression) else '(%r)'%den
		return '%s/%s' % (num_str, den_str)
	def __eq__(self, E):
		return isinstance(E, self.__class__) and (self._terms[0]==E._terms[0]) and (self._terms[1]==E._terms[1])

class SumExpression(CommutativeExpression):
	def __init__(self, expressions):
		super().__init__()
		self._terms = list(expressions)
	def __repr__(self):
		string = '%r' % self._terms[0]
		for t in self._terms[1:]:
			string += '%r'%t if isinstance(t, NegativeExpression) else '+%r'%t
		return string
	def __eq__(self, E):
		if super().__eq__(E):
			return True
		if isinstance(E, NegativeExpression):
			return E == RVFuncs.negative(SumExpression([ RVFuncs.negative(e) for e in self._terms ]))
		return False

class AndExpression(CommutativeExpression):
	def __init__(self, comparisons):
		super().__init__()
		self._terms = list(comparisons)
		self.name = ','.join('%s'%c.name for c in comparisons)
	def __repr__(self):
		return ','.join(['%r' % t for t in self._terms])
	def __eq__(self, E):
		return super().__eq__(E)


class ProductExpression(CommutativeExpression):
	def __init__(self, expressions):
		super().__init__()
		self._terms = list(expressions)
	def __repr__(self):
		string = '(%r)'%self._terms[0] if (isinstance(self._terms[0], SumExpression) and len(self._terms) > 1) else '%r'%self._terms[0]
		for t in self._terms[1:]:
			string += '*(%r)'%t if isinstance(t, SumExpression) else '*%r'%t
		return string
	
class MaxExpression(CommutativeExpression, SingleTermExpression):
	def __init__(self, expressions):
		super().__init__()
		self._terms = list(expressions)
	def __repr__(self):
		return 'MAX{%s}' % ', '.join([ '%r'%t for t in self._terms ])

class MaxRecipExpression(Expression, SingleTermExpression):
	def __init__(self, expression):
		super().__init__()
		self._terms = [expression]
	def __repr__(self):
		return 'MAX{%s, %s}' % (self._terms[0], RVFuncs.fraction(RVFuncs.constant(1), self._terms[0]))



# Functions to handle math operations safety when nans/infs are present

def safesum(a, b):
	a_inf, a_nan = np.isinf(a), np.isnan(a)
	b_inf, b_nan = np.isinf(b), np.isnan(b)
	if (a_nan or b_nan):
		return np.nan
	if a_inf and b_inf and (np.sign(a) != np.sign(b)):
		return np.nan
	return a + b

def safeprod(a, b):
	a_inf, a_nan = np.isinf(a), np.isnan(a)
	b_inf, b_nan = np.isinf(b), np.isnan(b)
	if (a_nan or b_nan):
		return np.nan
	if (a_inf and b==0) or (b_inf and a==0):
		return 0.0
	return a * b

def safediv(a, b):
	a_inf, a_nan = np.isinf(a), np.isnan(a)
	b_inf, b_nan = np.isinf(b), np.isnan(b)
	if (a_nan or b_nan) or (a_inf and b_inf):
		return np.nan
	if (b==0):
		return np.nan
	return a / b


class ConstraintManager():

	#Grammar specification
	grammar = r"""
		expr
		: term
		| expr '+' term
		| expr '-' term
		;

		exprs
		: expr
		| exprs ',' expr
		;

		term
		: unary
		| term '*' unary
		| term '/' unary
		;

		unary
		: primary
		| '|' expr '|'
		| '+' unary
		| '-' unary
		| 'max(' exprs ')'
		| 'min(' exprs ')'
		;

		primary
		: expected_value
		| number
		| '(' expr ')'
		;




		inner_expr
		: inner_term
		| inner_expr '+' inner_term
		| inner_expr '-' inner_term
		;

		inner_term
		: inner_unary
		| inner_term '*' inner_unary
		| inner_term '/' inner_unary
		;

		inner_unary
		: inner_primary
		| '|' inner_expr '|'
		| '+' inner_unary
		| '-' inner_unary
		;

		inner_primary
		: expected_value
		| number
		| variable
		| '(' inner_expr ')'
		;

		expected_value
		: 'E[' sample_set ']'
		;

		sample_set
		: comparison_list
		| comparison_list '|' comparison_list
		| inner_expr 
		| inner_expr '|' comparison_list
		;

		comparison_list
		: comparison
		| comparison_list ',' comparison
		;

		comparison
		: inner_expr inequality inner_expr
		| inner_expr equality inner_expr
		| inner_expr inequality equality inner_expr
		;

		terminals
		variable: /[a-zA-Z_$][a-zA-Z_$0-9]*/;
		inequality: /[<|>|!]/;
		equality: /[=]/;
		number: /\d+(\.\d+)?/;
	"""

	# Defines the composition rules for the grammar expressions
	actions = {
		"expr":   [  lambda _, nodes: nodes[0],
				     lambda _, nodes: RVFuncs.sum(nodes[0], nodes[2]),
				     lambda _, nodes: RVFuncs.sum(nodes[0], RVFuncs.negative(nodes[2]))],
		"exprs":  [  lambda _, nodes: nodes[0],
				     lambda _, nodes: ([*nodes[0], nodes[2]] if isinstance(nodes[0], list) else [nodes[0],nodes[2]])],
		"term":   [  lambda _, nodes: nodes[0],
				     lambda _, nodes: RVFuncs.product(nodes[0], nodes[2]),
				     lambda _, nodes: RVFuncs.fraction(nodes[0], nodes[2])],
	    "unary":  [  lambda _, nodes: nodes[0],
				     lambda _, nodes: RVFuncs.abs(nodes[1]),
	    			 lambda _, nodes: nodes[1],
	    			 lambda _, nodes: RVFuncs.negative(nodes[1]),
				     lambda _, nodes: RVFuncs.max(*nodes[1]), 
				     lambda _, nodes: RVFuncs.min(*nodes[1])], 
	    "primary": [ lambda _, nodes: nodes[0],
	    			 lambda _, nodes: RVFuncs.constant(nodes[0]),
	    			 lambda _, nodes: nodes[1]],	
		"inner_expr":   [  lambda _, nodes: nodes[0],
				     lambda _, nodes: RVFuncs.sum(nodes[0], nodes[2]),
				     lambda _, nodes: RVFuncs.sum(nodes[0], RVFuncs.negative(nodes[2]))],
		"inner_term":   [  lambda _, nodes: nodes[0],
				     lambda _, nodes: RVFuncs.product(nodes[0], nodes[2]),
				     lambda _, nodes: RVFuncs.fraction(nodes[0], nodes[2])],
	    "inner_unary":  [  lambda _, nodes: nodes[0],
				     lambda _, nodes: RVFuncs.abs(nodes[1]),
	    			 lambda _, nodes: nodes[1],
	    			 lambda _, nodes: RVFuncs.negative(nodes[1])], 
	    "inner_primary": [ lambda _, nodes: nodes[0],
	    			 lambda _, nodes: RVFuncs.constant(nodes[0]),
	    			 lambda _, nodes: RVFuncs.variable(nodes[0]),
	    			 lambda _, nodes: nodes[1]],	
	    "expected_value": [
	    			 lambda _, nodes: RVFuncs.expected_value(nodes[1])],
	    "sample_set": [
	    			 lambda _, nodes: RVFuncs.sample_set(RVFuncs.logical_and(nodes[0])),
	    			 lambda _, nodes: RVFuncs.sample_set(RVFuncs.logical_and(nodes[0]), RVFuncs.logical_and(nodes[2])),
	    			 lambda _, nodes: RVFuncs.sample_set(nodes[0]),
	    			 lambda _, nodes: RVFuncs.sample_set(nodes[0], RVFuncs.logical_and(nodes[2]))],
	    "comparison_list": [
	    			 lambda _, nodes: nodes,
	    			 lambda _, nodes: [ *nodes[0], nodes[2] ]],
	    "comparison": [
	    			 lambda _, nodes: RVFuncs.comparator_variable(nodes[0], nodes[1], nodes[2]),
	    			 lambda _, nodes: RVFuncs.comparator_variable(nodes[0], nodes[1], nodes[2]),
	    			 lambda _, nodes: RVFuncs.comparator_variable(nodes[0], nodes[1]+nodes[2], nodes[3])],
	    "number":    lambda _, value: value,
	    "variable":  lambda _, value: value,
	}

	def parse(self, string, debug=False):
		string = replace_keywords(string,self.keywords)
		g = Grammar.from_string(ConstraintManager.grammar)
		parser = Parser(g, debug=debug, actions=ConstraintManager.actions)
		return parser.parse(string)

	def __init__(self, defined_variables, constraints, trivial_bounds={}, keywords={}):
		self.keywords = keywords
		g = Grammar.from_string(ConstraintManager.grammar)
		constraints = [ replace_keywords(const,keywords) for const in constraints]
		parser = Parser(g, debug=False, actions=ConstraintManager.actions)
		self.defined_variables = defined_variables
		self.constraint_strs = constraints
		self.n_constraints = len(constraints)
		self.constraint_exprs = [ parser.parse(c) for c in constraints ]
		self.trivial_bounds = { v:(-np.inf,np.inf) for v in self.defined_variables }
		self.trivial_bounds.update(trivial_bounds)
		self.identify_base_variables()
		self.trivial_bounds.update({ ev.expression.name:(0,1) for ev in self.expected_values.values() if isinstance(ev.expression, ComparatorExpression)})
		assert self.base_variables.issubset(defined_variables), 'ConstraintManager: Constraints depened on undefined variables. Defined variables are: %r' % defined_variables
		assert self.cond_variables.issubset(defined_variables), 'ConstraintManager: Constraints depened on undefined variables. Defined variables are: %r' % defined_variables
		self.values = { n:None for n in self.defined_variables }


	def identify_base_variables(self):
		# Identifies unique identifier variables in all constraints by their name
		self.base_variables = set()
		self.base_variables_per_constraint = []
		self.cond_variables = set()
		self.cond_variables_per_constraint = []
		self.expected_values = dict()
		self.expected_values_per_constraint = []
		for E in self.constraint_exprs:
			bvars, cvars, evars = self._identify_base_vars(E)
			self.base_variables_per_constraint.append(bvars)
			self.base_variables = self.base_variables.union(bvars)
			self.cond_variables_per_constraint.append(cvars)
			self.cond_variables = self.cond_variables.union(cvars)
			self.expected_values_per_constraint.append(evars)
			self.expected_values.update(evars)

	@staticmethod
	def _identify_base_vars(E):
		# Recursively identifies unique variables in <E> by their name
		if isinstance(E, ExpectedValue):
			S, C, _ = ConstraintManager._identify_base_vars(E.sample_set)
			return S, C, {E.name:E.sample_set}
		if isinstance(E, SampleSet):
			S, C, B = ConstraintManager._identify_base_vars(E.expression)
			if not(E.condition is None):
				C, _, _ = ConstraintManager._identify_base_vars(E.condition)
			return S, C, B
		if isinstance(E, ComparatorExpression):
			return ConstraintManager._identify_base_vars(E.variable)
		if isinstance(E, VariableExpression):
			return set([E.name]), set(), dict()
		if isinstance(E, ConstantExpression):
			return set(), set(), dict()
		base_vars, cond_vars, ev_vars = set(), set(), dict()
		for _E in E._terms:
			S, C, B = ConstraintManager._identify_base_vars(_E)
			base_vars = base_vars.union(S)
			cond_vars = cond_vars.union(C)
			ev_vars.update(B)
		return base_vars, cond_vars, ev_vars

	def set_data(self, values):
		# Sets defined variables to have the values in <values>
		for n in self.defined_variables:
			if n in values.keys():
				self.values[n] = values[n]

	def has_defined_values(self):
		# Returns True iff all defined variables have non-None values set
		return not(any( self.values[v] is None for v in self.base_variables ))



	def evaluate(self):
		# Computes the value of each constraint expression given data set by set_data()
		assert self.has_defined_values(), 'ConstraintManager.evaluate(): Undefined values %r' % [ k for k,v in self.values.items() if v is None ]
		return np.array([ self._evaluate(E) for E in self.constraint_exprs ])

	def _evaluate(self, E):
		# Recursively evaluates expression <E> using data set by set_data()
		if isinstance(E, ConstantExpression):
			return E.value
		if isinstance(E, ExpectedValue):
			with warnings.catch_warnings():
				warnings.filterwarnings('error')
				try:
					return self._evaluate(E.sample_set).mean()
				except RuntimeWarning:
					return np.nan
		if isinstance(E, SampleSet):
			values = self._evaluate(E.expression)
			if not(E.condition is None):
				cvalues = self._evaluate(E.condition)
				values = values[cvalues]
			return values
		if isinstance(E, ComparatorExpression):
			values1 = self._evaluate(E.variable)
			values2 = self._evaluate(E.value)
			return COMPARATORS[E.comparator](values1, values2)
		if isinstance(E, VariableExpression):
			return self.values[E.name]
		if isinstance(E, NegativeExpression):
			return -self._evaluate(E._terms[0])
		if isinstance(E, AbsExpression):
			return np.abs(self._evaluate(E._terms[0]))
		if isinstance(E, SumExpression):
			return np.sum([ self._evaluate(_E) for _E in E._terms ])
		if isinstance(E, ProductExpression):
			return np.prod([ self._evaluate(_E) for _E in E._terms ])
		if isinstance(E, AndExpression):
			values = self._evaluate(E._terms[0])
			for _E in E._terms[1:]:
				values = np.logical_and(values, self._evaluate(_E))
			return values
		if isinstance(E, FractionExpression):
			v_num = self._evaluate(E._terms[0]) 
			v_den = self._evaluate(E._terms[1])
			return safediv(v_num, v_den)
		if isinstance(E, MaxExpression):
			return np.max([ self._evaluate(_E) for _E in E._terms ])
		if isinstance(E, MaxRecipExpression):
			_E = E._terms[0]
			if isinstance(_E, FractionExpression):
				v_num = self._evaluate(_E._terms[0])
				v_den = self._evaluate(_E._terms[1])
				vs = [safediv(v_num, v_den), safediv(v_den, v_num)]
			else:
				v  = self._evaluate(_E)
				vs = [v, safediv(1, v)]
			if all(np.isnan(vs)):
				return np.nan
			return np.nanmax(vs)

	def upper_bound_constraints(self, all_deltas, mode='hoeffding', interval_scaling=1.0, n_scale=1.0, term_values={}):
		constraint_bounds = self.bound_constraints(all_deltas, mode=mode, interval_scaling=interval_scaling, n_scale=n_scale, term_values=term_values)
		return np.array([ b[1] for b in constraint_bounds ])

	def bound_constraints(self, all_deltas, mode='hoeffding', interval_scaling=1.0, n_scale=1.0, term_values={}):
		assert self.has_defined_values(), 'ConstraintManager.bound(): Undefined values %r' % [ k for k,v in self.values.items() if v is None ]
		deltas   = { name : None for name in self.expected_values }
		bounds   = { name : None for name in self.expected_values }
		constraint_bounds = []
		
		for cnum, (E,delta_tot) in enumerate(zip(self.constraint_exprs,all_deltas)):
			# Bound the base variables needed for this constraint
			variables = self.expected_values_per_constraint[cnum]
			delta_per_var = delta_tot / len(variables)
			
			for name in variables:
				if not(deltas[name] is None) and (deltas[name] == delta_per_var):
					bounds[name] = bounds_last[name]
				else:
					bounds[name] = self.bound_variable(name, delta_per_var, mode=mode, n_scale=n_scale)
					deltas[name] = delta_per_var
			# Bound the expression for this constraint
			l, u = ConstraintManager.bound_expression(E, bounds={v:bounds[v] for v in variables})
			# Inflate the bound if needed
			if not(any(np.isinf([l,u])) or any(np.isnan([l,u]))):
				mod = 0.5*(u-l)*(interval_scaling-1)
				l, u = l-mod, u+mod
			constraint_bounds.append((l,u))
		return constraint_bounds

	def bound_expression(E, bounds):
		if isinstance(E, ConstantExpression):
			return (E.value, E.value)
		if isinstance(E, ExpectedValue):
			return bounds[E.name]
		if isinstance(E, NegativeExpression):
			l, u = ConstraintManager.bound_expression(E._terms[0], bounds)
			return (-u, -l)
		if isinstance(E, AbsExpression):
			l, u = ConstraintManager.bound_expression(E._terms[0], bounds)
			if l <= 0 and u >= 0:
				return (0, max(u,-l))
			if u < 0:
				return (-u, -l)
			if l > 0:
				return (l, u)
		if isinstance(E, SumExpression):
			l, u = 0, 0
			for _E in E._terms:
				_l, _u = ConstraintManager.bound_expression(_E, bounds)
				l, u = safesum(l,_l), safesum(u,_u)
			return (l, u)
		if isinstance(E, ProductExpression):
			l, u = ConstraintManager.bound_expression(E._terms[0], bounds)
			for _E in E._terms[1:]:
				ln, un = ConstraintManager.bound_expression(_E, bounds)
				cll, clu = safeprod(l,ln), safeprod(l,un)
				cul, cuu = safeprod(u,ln), safeprod(u,un)
				interval_corners = [cll, clu, cul, cuu]
				l, u = min(interval_corners), max(interval_corners)
			return (l,u)
		if isinstance(E, FractionExpression):
			ln, un = ConstraintManager.bound_expression(E._terms[0], bounds)
			ld, ud = ConstraintManager.bound_expression(E._terms[1], bounds)
			# If ln = un = ld = ud = 0, we return (-inf,inf) which is a useless bound, instead of (nan,nan)
			# Values are based on treating the input intervals as open intervals
			# If an interval is empty (l==u), it is treated as an infinitesimal interval (-e+l,l+e) instead
			if (ld == 0) and (ud == 0):
				return (-np.inf, np.inf)
			if (ld == 0 or ud == 0) and (ln <= 0) and (un >= 0):
				return (-np.inf, np.inf)
			if (ld == 0) and (ln == 0):
				return (np.inf, np.inf)
			if (ld == 0 or ud == 0) and (un <= 0):
				return (np.inf, np.inf)
			cll, clu = safediv(ln,ld), safediv(ln,ud)
			cul, cuu = safediv(un,ld), safediv(un,ud)
			interval_corners = [cll, clu, cul, cuu]
			return min(interval_corners), max(interval_corners)
		if isinstance(E, MaxExpression):
			l, u = ConstraintManager.bound_expression(E._terms[0], bounds)
			for _E in E._terms[1:]:
				ln, un = ConstraintManager.bound_expression(_E, bounds)
				l, u = max(l,ln), max(u,un)
			return (l,u)
		if isinstance(E, MaxRecipExpression):
			l, u = ConstraintManager.bound_expression(E._terms[0], bounds)
			li = 1/l if not(l==0) else np.inf
			ui = 1/u if not(u==0) else np.inf

			if any(np.isnan([l,u])):
				return (np.nan, np.nan)

			elif l >= 1 and u >= 1:
				return (l, u)
			elif l >= 0 and u >= 1:
				return (1, max(u,li))
			elif l >= -1 and u >= 1:
				return (l, np.inf)
			elif l < -1 and u >= 1:
				return (-1, np.inf)

			elif l >= 0 and u >= 0:
				return (ui, li)
			elif l >= -1 and u >= 0:
				return (l, np.inf)
			elif l < -1 and u >= 0:
				return (-1, np.inf)

			elif l >= -1 and u >= -1:
				return (l, u)
			elif l < -1 and u >= -1:
				return (-1, max(li, u))

			elif l < -1 and u < -1:
				return (ui, li)

	def bound_variable(self, name, delta, mode='hoeffding', n_scale=1.0, bootstrap_samples=1000):
		# Returns a <delta>-probability confidence interval on the value of <name> using <mode>.

		mode = mode.lower()
		if isinstance(self.expected_values[name].expression, AndExpression):
			a, b = (0,1)
		else:
			a, b = self.trivial_bounds[self.expected_values[name].expression.name]

		# "Computes" the trivial bounds
		if mode == 'trivial':
			return (a, b)
	
		# Get the sample(s) associated with <name>
		S = self._evaluate(self.expected_values[name])
		try:
			n = len(S)
		except TypeError:
			S = np.array([ S ])
			n = len(S)

		#### Bounds below this point require at least one sample ####

		if (n == 0 or n_scale == 0):
			return (a, b)

		# Now that we know the mean is well-defined, compute it
		mean = np.mean(S)
		n_scaled = n * n_scale

		# Computes the hoeffding bound
		if mode == 'hoeffding':
			offset = (b-a) * np.sqrt(0.5*np.log(2/delta)/n_scaled)
			l, u = safesum(mean,-offset), safesum(mean,offset)
			return (max(a,l), min(b,u))

		# Computes the bootstrap bound
		if mode == 'bootstrap':
			B = np.random.multinomial(n, np.ones(n)/n, bootstrap_samples)
			Z = (B*S[None,:]).mean(1)
			l, u = np.percentile(Z, (100*delta/2, 100*(1-delta/2)))
			return (max(a,l), min(b,u))

		#### Bounds below this point require at least two samples ####
		
		if len(S) == 1:
			return (a, b)
	
		# Now that we know the standard deviation is well-defined, compute it
		std = np.std(S,ddof=1)
		
		# If standard deviation too close to zero, apply the rule of three
		if np.isclose(std, 0.0):
			if np.isclose(mean, a):
				return (a, (b-a)*3.0/n_scaled)
			elif np.isclose(mean, b):
				return ((b-a)*(1-3.0/n_scaled)+a, b)
			return (a, b)

		# Computes the t-test inversion bound 	
		if mode == 'ttest': 
			offset = std * t.ppf(1-delta/2,n-1) / np.sqrt(n-1)
			l, u = safesum(mean,-offset), safesum(mean,offset)
			return (max(a,l), min(b,u))

		# Should never reach here, so return trivial bounds
		return (a, b)
		


def _replace_keyword(s, name, replacement): 
	return replacement.join(re.split(r'(?<![a-zA-Z_])%s(?![a-zA-Z_\(])' % name, s))                                                                                                                                                                             

def replace_keywords(s, repls): 
	for name, repl in repls.items(): 
		s = _replace_keyword(s, name, repl) 
		if re.match(r'^E\[.*\]$', repl):
			conditions = re.findall('(?<![a-zA-Z_])%s\(([^\)]+)\)[^a-zA-Z_]*' % name, s)
			s, *rest   = re.split(r'(?<![a-zA-Z_])%s\([^\)]*\)(?![a-zA-Z_])' % name, s)
			for c, r in zip(conditions, rest):
				s += ('%s,%s]%s' if ('|' in repl) else '%s|%s]%s') % (repl[:-1],c,r)
	return s 