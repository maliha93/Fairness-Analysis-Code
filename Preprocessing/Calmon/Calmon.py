# Load all necessary packages
import sys
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, CreditDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from common_utils import compute_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from IPython.display import Markdown, display
import time 
from metric import metric

def Adult(fname=None):
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    dataset_orig = AdultDataset(protected_attribute_names=['sex'], privileged_classes=[['Male']],\
                   categorical_features=['age', 'edu_level'], features_to_keep=['age', 'edu_level', 'sex'], fname=fname)
    
    optim_options = {
        "distortion_fun": get_distortion_adult,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }

    #random seed
    np.random.seed(1)
    # Split into train and test
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    #print(dataset_orig_train)
    #return
    OP = OptimPreproc(OptTools, optim_options,
                      unprivileged_groups = unprivileged_groups,
                      privileged_groups = privileged_groups)

    OP = OP.fit(dataset_orig_train)
    # Transform training data and align features
    dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)
    dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)
    dataset_orig_test = dataset_transf_train.align_datasets(dataset_orig_test)
    dataset_transf_test = OP.transform(dataset_orig_test, transform_Y = True)
    dataset_transf_test = dataset_orig_test.align_datasets(dataset_transf_test)
    
    ttrain, ttest = AdultDataset().split([0.7], shuffle=False)
    ttrain, _ = ttrain.convert_to_dataframe(de_dummy_code=True)
    ttest, _ = ttest.convert_to_dataframe(de_dummy_code=True)
    ttest.to_csv("results_Calmon/adult_test_notrepaired.csv", index=False)
    
    train, _ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True)
    test, _ = dataset_transf_test.convert_to_dataframe(de_dummy_code=True)
    ttrain[['age', 'edu_level', 'sex', 'income']] = train[['age', 'edu_level', 'sex', 'income']]
    ttest[['age', 'edu_level', 'sex', 'income']] = ttest[['age', 'edu_level', 'sex', 'income']]
    ttrain.to_csv("results_Calmon/adult_train_repaired.csv", index=False)
    ttest.to_csv("results_Calmon/adult_test_repaired.csv", index=False)
        
    
def Compas(f1='', f2='', fname=None):
    privileged_groups = [{'Race': 1}]
    unprivileged_groups = [{'Race': 0}]
    dataset_orig = CompasDataset(fname=fname)
    #print(dataset_orig)

    optim_options = {
        "distortion_fun": get_distortion_compas,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }

    #random seed
    np.random.seed(1)
    # Split into train and test
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    OP = OptimPreproc(OptTools, optim_options,
                      unprivileged_groups = unprivileged_groups,
                      privileged_groups = privileged_groups)

    OP = OP.fit(dataset_orig_train)

    # Transform training data and align features
    dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)
    dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)
    dataset_orig_test = dataset_transf_train.align_datasets(dataset_orig_test)
    dataset_transf_test = OP.transform(dataset_orig_test, transform_Y = True)
    dataset_transf_test = dataset_orig_test.align_datasets(dataset_transf_test)
    
    train, _ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True)
    test, _ = dataset_transf_test.convert_to_dataframe(de_dummy_code=True)
    train.to_csv(f1+"results_Calmon/compas_train_repaired"+f2+".csv", index=False)
    test.to_csv(f1+"results_Calmon/compas_test_repaired"+f2+".csv", index=False)
    dataset_orig_test, _ = dataset_orig_test.convert_to_dataframe(de_dummy_code=True)
    dataset_orig_test.to_csv(f1+"results_Calmon/compas_test_notrepaired"+f2+".csv", index=False)
    
def German(fname=None):
    
    privileged_groups = [{'Sex': 1}]
    unprivileged_groups = [{'Sex': 0}]
    dataset_orig = GermanDataset(protected_attribute_names=['Sex'], privileged_classes=[['Male']],\
                    categorical_features=[ 'Status', 'Credit_history', 'Savings', 'Age'],\
                    features_to_keep=['Status', 'Credit_history', 'Savings', 'Age'],\
                    features_to_drop=['Month', 'Credit_amount', 'Investment', 'Property', 'Housing'], fname=fname)
   # dataset_orig = load_preproc_data_german(['Sex'])
    
    optim_options = {
        "distortion_fun": get_distortion_german,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }

    #random seed
    np.random.seed(1)
    # Split into train and test
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    OP = OptimPreproc(OptTools, optim_options,
                      unprivileged_groups = unprivileged_groups,
                      privileged_groups = privileged_groups)

    OP = OP.fit(dataset_orig_train)

    # Transform training data and align features
    dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)
    dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)
    dataset_orig_test = dataset_transf_train.align_datasets(dataset_orig_test)
    dataset_transf_test = OP.transform(dataset_orig_test, transform_Y = True)
    dataset_transf_test = dataset_orig_test.align_datasets(dataset_transf_test)
    
    ttrain, ttest = GermanDataset().split([0.7], shuffle=False)
    ttrain, _ = ttrain.convert_to_dataframe(de_dummy_code=True)
    ttest, _ = ttest.convert_to_dataframe(de_dummy_code=True)

    train, _ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True)
    test, _ = dataset_transf_test.convert_to_dataframe(de_dummy_code=True)
    ttrain[['Status', 'Credit_history', 'Savings', 'Age', 'Sex', 'credit']] = train[['Status',\
                                        'Credit_history', 'Savings', 'Age', 'Sex', 'credit']]
    ttest[['Status', 'Credit_history', 'Savings', 'Age', 'Sex', 'credit']] = ttest[['Status',\
                                        'Credit_history', 'Savings', 'Age', 'Sex', 'credit']]
    ttrain.to_csv("results_Calmon/german_train_repaired.csv", index=False)
    ttest.to_csv("results_Calmon/german_test_repaired.csv", index=False)
    
    _, dataset_orig_test = GermanDataset().split([0.7], shuffle=False)
    dataset_orig_test, _ = dataset_orig_test.convert_to_dataframe(de_dummy_code=True)
    dataset_orig_test.to_csv("results_Calmon/german_test_notrepaired.csv", index=False)

def Calmon(dataset, fname=None):
    
    if dataset == 'adult':
        Adult(fname)
    elif dataset == 'compas':
        Compas(fname=fname)
    elif dataset == 'german':
        German(fname)
    elif dataset == 'credit':
        Credit(fname)



