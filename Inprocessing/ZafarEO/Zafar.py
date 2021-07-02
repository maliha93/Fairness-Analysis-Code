from __future__ import division
#import urllib2
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle
import utils as ut
import funcs_disp_mist as fdm
import time

SEED = 1234
seed(SEED)
np.random.seed(SEED)



def Zafar(dataset):
    if dataset == 'adult':
        test_adult_data()
    elif dataset == 'compas':
        test_compas_data()
    else:
        test_german_data()
    
    return


def load_adult_data(load_data_size=None):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    attrs = ['age', 'workclass', 'edu_level', 'marital_status', 'occupation', 'relationship', 
             'race', 'sex', 'hours_per_week', 'native_country'] # all attributes 
    int_attrs = ['age', 'edu_level', 'hours_per_week'] # attributes with integer values -- the rest are categorical
    
    sensitive_attrs = ['sex'] # the fairness constraints will be used for this feature
    
    attrs_to_ignore = ['sex', 'race'] # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
    
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    data_files = ['dataset/Adult.csv']#["adult.data", "adult.test"]



    X = []
    y = []
    x_control = {}
    
    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []
            

    for f in data_files:
    
        for line in open(f):
            line = line.strip()
            if line.startswith("age") or line == "":
                continue
            line = line.split(",")
            if "?" in line: # if a line has missing attributes, ignore it
                continue
            
            class_label = line[-1]
            if class_label in ["<=50K.", "<=50K", "0"]:
                class_label = -1
            elif class_label in [">50K.", ">50K", "1"]:
                class_label = +1
            else:
                raise Exception("Invalid class label value")
            y.append(class_label)


            for i in range(0,len(line)-1):
                attr_name = attrs[i]
                attr_val = line[i]
                # reducing dimensionality of some very sparse features
                if attr_name == "native_country":
                    if attr_val!="United-States":
                        attr_val = "Non-United-Stated"
                elif attr_name == "education":
                    if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                        attr_val = "prim-middle-school"
                    elif attr_val in ["9th", "10th", "11th", "12th"]:
                        attr_val = "high-school"

                if attr_name in sensitive_attrs:
                    x_control[attr_name].append(attr_val)
                elif attr_name in attrs_to_ignore:
                    pass
                else:
                    attrs_to_vals[attr_name].append(attr_val)

    def convert_attrs_to_ints(d): # discretize the string attributes
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs: continue
            uniq_vals = sorted(list(set(attr_vals))) # get unique values

            # compute integer codes for the unique values
            val_dict = {}
            for i in range(0,len(uniq_vals)):
                val_dict[uniq_vals[i]] = i

            # replace the values with their integer encoding
            for i in range(0,len(attr_vals)):
                attr_vals[i] = val_dict[attr_vals[i]]
            d[attr_name] = attr_vals

    
    # convert the discrete values to their integer representations
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)
    

    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country" or attr_name == "sex": 
            # the way we encoded native country, its binary now so no need to apply one hot encoding on it
            X.append(attr_vals)

        else: 
            attr_vals, index_dict = ut.get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:                
                X.append(inner_col) 
                

    # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype = float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)
        
        
    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    #shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print ("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    return X, y, x_control
    


def test_adult_data():

	data_type = 1
	X, y, x_control = load_adult_data()
	sensitive_attrs = x_control.keys()
    
	""" Split the data into train and test """
	train_fold_size = 0.7
	split_point = int(round(float(X.shape[0]) * train_fold_size))
	x_train, y_train, x_control_train, x_test, y_test, x_control_test, _= \
    ut.split_into_train_test(X, y, x_control, train_fold_size)

	cons_params = None # constraint parameters, will use them later
	loss_function = "logreg" # perform the experiments with logistic regression
	EPS = 1e-6

	def train_test_classifier():
		w = fdm.train_model_disp_mist(x_train, y_train, x_control_train, loss_function, EPS, cons_params)
		train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test, y_pred =\
        fdm.get_clf_stats(w, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs)
		return w, test_score, s_attr_to_fp_fn_test, y_pred

    
	""" Classify the data while optimizing for accuracy """
	print()
	print ("== Unconstrained (original) classifier ==")
	w_uncons, acc_uncons, s_attr_to_fp_fn_test_uncons, _ = train_test_classifier()
	print ("\n-----------------------------------------------------------------------------------\n")

	""" Now classify such that we optimize for accuracy while achieving perfect fairness """
	print()
	start = time.time()
	print ("\n\n== Equalized Odds ==")	# setting parameter for constraints
	cons_type = 4 # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
	tau = 5.0
	mu = 1.2
	sensitive_attrs_to_cov_thresh = {"sex": {0:{0:0, 1:0}, 1:{0:0, 1:0}, 2:{0:0, 1:0}}} # zero covariance threshold, means try to get the fairest solution
	cons_params = {"cons_type": cons_type, 
					"tau": tau, 
					"mu": mu, 
					"sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

	w_cons, acc_cons, s_attr_to_fp_fn_test_cons, y_pred  = train_test_classifier()
	print ("\n-----------------------------------------------------------------------------------\n")
	end = time.time()
	print("Time: ", end - start)
	df = pd.read_csv("dataset/Adult.csv")
	df = df.iloc[split_point:, :]
	y_pred[y_pred < 0] = 0
	df['pred'] = y_pred
	df.to_csv("results_zafarEO/adult_test_repaired.csv", index=False)
	return


def load_german_data(load_data_size=None):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    attrs = ['Month', 'Credit_amount', 'Investment', 'Age','Sex', 'Status', 'Credit_history',\
             'Savings', 'Property', 'Housing'] # all attributes
    
    int_attrs = ['Month', 'Credit_amount', 'Investment', 'Age'] # attributes with integer values -- the rest are categorical
    
    sensitive_attrs = ['Sex'] # the fairness constraints will be used for this feature
    attrs_to_ignore = ['Sex'] 
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
    data_files = ["dataset/German.csv"]

    X = []
    y = []
    x_control = {}

    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []
            

    for f in data_files:
        for line in open(f):
            if line.startswith("Month"):
                continue
            line = line.strip()
            if line == "": continue # skip empty lines
            line = line.split(",")
            #print(line)
            if "?" in line: # if a line has missing attributes, ignore it
                continue
            
            class_label = line[-1]
            #print(class_label)
            if class_label in ["Bad", "0"]:
                class_label = -1
            elif class_label in ["good", "1"]:
                class_label = +1
            else:
                raise Exception("Invalid class label value")

            y.append(class_label)

            for i in range(0,len(line)-1):
                attr_name = attrs[i]
                attr_val = line[i]
            
                if attr_name in sensitive_attrs:
                    x_control[attr_name].append(attr_val)
                elif attr_name in attrs_to_ignore:
                    pass
                else:
                    attrs_to_vals[attr_name].append(attr_val)
    
    def convert_attrs_to_ints(d): # discretize the string attributes
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs: continue
            uniq_vals = sorted(list(set(attr_vals))) # get unique values

            # compute integer codes for the unique values
            val_dict = {}
            for i in range(0,len(uniq_vals)):
                val_dict[uniq_vals[i]] = i

            # replace the values with their integer encoding
            for i in range(0,len(attr_vals)):
                attr_vals[i] = val_dict[attr_vals[i]]
            d[attr_name] = attr_vals

    
    # convert the discrete values to their integer representations
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)
    
    

    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "Sex": # the way we encoded native country, its binary now so no need to apply one hot encoding on it
            X.append(attr_vals)

        else: 
            attr_vals, index_dict = ut.get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:                
                X.append(inner_col) 
                
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype = float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)
        
        
    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    #shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print ("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    return X, y, x_control
   

def test_german_data():
		
	X, y, x_control = load_german_data()
	sensitive_attrs = x_control.keys()
	
	""" Split the data into train and test """
	train_fold_size = 0.7
	split_point = int(round(float(X.shape[0]) * train_fold_size))
	x_train, y_train, x_control_train, x_test, y_test, x_control_test, _ = \
    ut.split_into_train_test(X, y, x_control, train_fold_size)
	cons_params = None # constraint parameters, will use them later
	loss_function = "logreg" # perform the experiments with logistic regression
	EPS = 1e-6

	def train_test_classifier():
		w = fdm.train_model_disp_mist(x_train, y_train, x_control_train, loss_function, EPS, cons_params)
		train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test, y_pred =\
        fdm.get_clf_stats(w, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs)
		
		# accuracy and FPR are for the test because we need of for plotting
		return w, test_score, s_attr_to_fp_fn_test, y_pred
		

	""" Classify the data while optimizing for accuracy """
	print()
	print ("== Unconstrained (original) classifier ==")
	w_uncons, acc_uncons, s_attr_to_fp_fn_test_uncons, _ = train_test_classifier()
	print ("\n-----------------------------------------------------------------------------------\n")

	""" Now classify such that we optimize for accuracy while achieving perfect fairness """
	
	print()
	start = time.time()
	print ("\n\n== Equalized Odds ==")	# setting parameter for constraints
	cons_type = 4 # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
	tau = 5.0
	mu = 1.2
	sensitive_attrs_to_cov_thresh = {"Sex": {0:{0:0, 1:0}, 1:{0:0, 1:0}, 2:{0:0, 1:0}}} # zero covariance threshold, means try to get the fairest solution
	cons_params = {"cons_type": cons_type, 
					"tau": tau, 
					"mu": mu, 
					"sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

	w_cons, acc_cons, s_attr_to_fp_fn_test_cons, y_pred  = train_test_classifier()
	end = time.time()
	print("Time: ", end - start)
	print ("\n-----------------------------------------------------------------------------------\n")
	df = pd.read_csv("dataset/German.csv")
	df = df.iloc[split_point:, :]
	y_pred[y_pred < 0] = 0
	df['pred'] = y_pred
	df.to_csv("results_zafarEO/german_test_repaired.csv", index=False)
	return
    
    
def load_compas_data(load_data_size=None, f="dataset/Compas.csv"):

    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    attrs = ['Sex', 'Age', 'Race', 'Prior'] # all attributes
    int_attrs = ['Age', 'Sex', 'Prior'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['Race'] # the fairness constraints will be used for this feature
    attrs_to_ignore = ['Race'] 
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    data_files = [f]

    X = []
    y = []
    x_control = {}
    
    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    for f in data_files:
        
        for line in open(f):
            if line.startswith("Sex"):
                continue
            line = line.strip()
            if line == "": continue # skip empty lines
            line = line.split(",")
            
            if "?" in line: # if a line has missing attributes, ignore it
                continue
                
            
            class_label = line[-1]
            if class_label in [ "0", "0.0"]:
                class_label = -1
            elif class_label in [ "1", "1.0"]:
                class_label = +1
            else:
                print(class_label)
                raise Exception("Invalid class label value")
            y.append(class_label)

            for i in range(0,len(line)-1):
                attr_name = attrs[i]
                attr_val = line[i]
                
                if(attr_name == 'Sex'):
                    if(line[i] == 'Male'):
                        attr_val = '1'
                    else:
                        attr_val = '0'
                        
                if(attr_name == 'Race'):
                    if(line[i] == 'African-American'):
                        attr_val = 'African-American'
                    else:
                        attr_val = 'Others'

                if attr_name in sensitive_attrs:
                    x_control[attr_name].append(attr_val)
                elif attr_name in attrs_to_ignore:
                    pass
                else:
                    attrs_to_vals[attr_name].append(attr_val)

    def convert_attrs_to_ints(d): # discretize the string attributes
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs: continue
            uniq_vals = sorted(list(set(attr_vals))) # get unique values

            # compute integer codes for the unique values
            val_dict = {}
            for i in range(0,len(uniq_vals)):
                val_dict[uniq_vals[i]] = i

            # replace the values with their integer encoding
            for i in range(0,len(attr_vals)):
                attr_vals[i] = val_dict[attr_vals[i]]
            d[attr_name] = attr_vals

    
    # convert the discrete values to their integer representations
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)


    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs: # the way we encoded native country, its binary now so no need to apply one hot encoding on it
            X.append(attr_vals)

        else:            
            attr_vals, index_dict = ut.get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:                
                X.append(inner_col) 

    # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype = float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)
        
    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    #shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print ("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    return X, y, x_control
    
    
    

def test_compas_data(f="dataset/Compas.csv", f1='', f2=''):
	
	""" Generate the synthetic data """
	data_type = 1
	X, y, x_control = load_compas_data(f=f)
	sensitive_attrs = x_control.keys()

	train_fold_size = 0.7
	split_point = int(round(float(X.shape[0]) * train_fold_size))
	x_train, y_train, x_control_train, x_test, y_test, x_control_test, _= \
    ut.split_into_train_test(X, y, x_control, train_fold_size)
    
	cons_params = None # constraint parameters, will use them later
	loss_function = "logreg" # perform the experiments with logistic regression
	EPS = 1e-6
	def train_test_classifier():
		w = fdm.train_model_disp_mist(x_train, y_train, x_control_train, loss_function, EPS, cons_params)
		train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test, y_pred =\
        fdm.get_clf_stats(w, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs)
		return w, test_score, s_attr_to_fp_fn_test, y_pred
		

	""" Classify the data while optimizing for accuracy """
	print()
	print ("== Unconstrained (original) classifier ==")
	w_uncons, acc_uncons, s_attr_to_fp_fn_test_uncons, _ = train_test_classifier()
	print ("\n-----------------------------------------------------------------------------------\n")

	""" Now classify such that we optimize for accuracy while achieving perfect fairness """
	
	print()
	start = time.time()
	print ("\n\n== Equalized Odds ==")	# setting parameter for constraints
	cons_type = 2 # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
	tau = 5.0
	mu = 1.2
	sensitive_attrs_to_cov_thresh = {"Race": {0:{0:0, 1:0}, 1:{0:0, 1:0}, 2:{0:0, 1:0}}} # zero covariance threshold, means try to get the fairest solution
	cons_params = {"cons_type": cons_type, 
					"tau": tau, 
					"mu": mu, 
					"sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}

	w_cons, acc_cons, s_attr_to_fp_fn_test_cons, y_pred  = train_test_classifier()
	end = time.time()
	print("Time: ", end - start)
	print ("\n-----------------------------------------------------------------------------------\n")
	df = pd.read_csv(f)
	df = df.iloc[split_point:, :]
	y_pred[y_pred < 0] = 0
	df['pred'] = y_pred
	df.to_csv(f1+"results_zafarEO/compas_test_repaired"+f2+".csv", index=False)
   
