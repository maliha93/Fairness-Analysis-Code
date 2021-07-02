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
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from common_utils import compute_metrics
from sklearn.metrics import confusion_matrix
from metric import metric            

def Adult(fname=None):
    
    protected = 'sex'
    dataset_used = "adult" 
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    dataset_orig = load_preproc_data_adult(protected_attributes=['sex'], fname=fname)
    index = dataset_orig.feature_names.index(protected)
    np.random.seed(1)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                   privileged_groups=privileged_groups)
    RW.fit(dataset_orig_train)
    dataset_transf_train = RW.transform(dataset_orig_train)
    df, _ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True)
    df['weights'] = dataset_transf_train.instance_weights
    df.to_csv("results_Kamiran/adult_train_repaired.csv", index=False)
    df, _ = dataset_orig_test.convert_to_dataframe(de_dummy_code=True)
    df.to_csv("results_Kamiran/adult_test.csv", index=False)

    
def Compas(f1='', f2='', fname=None):    
    protected = 'Race'
    privileged_groups = [{'Race': 1}]
    unprivileged_groups = [{'Race': 0}]
    dataset_orig = load_preproc_data_compas(['Race'], fname=fname)
    index = dataset_orig.feature_names.index(protected)

    #random seed for calibrated equal odds prediction
    np.random.seed(1)

    # Get the dataset and split into train and test
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                   privileged_groups=privileged_groups)
    RW.fit(dataset_orig_train)
    dataset_transf_train = RW.transform(dataset_orig_train)
    df, _ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True)
    df['weights'] = dataset_transf_train.instance_weights
    df.to_csv(f1+"results_Kamiran/compas_train_repaired"+f2+".csv", index=False)
    df, _ = dataset_orig_test.convert_to_dataframe(de_dummy_code=True)
    df.to_csv(f1+"results_Kamiran/compas_test"+f2+".csv", index=False)


def German(fname=None):    
    protected = 'Sex'
    privileged_groups = [{'Sex': 1}]
    unprivileged_groups = [{'Sex': 0}]
    dataset_orig = load_preproc_data_german(['Sex'], fname=fname)
    index = dataset_orig.feature_names.index(protected)

    np.random.seed(1)
    # Get the dataset and split into train and test
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                   privileged_groups=privileged_groups)
    RW.fit(dataset_orig_train)
    dataset_transf_train = RW.transform(dataset_orig_train)
    df, _ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True)
    df['weights'] = dataset_transf_train.instance_weights
    df.to_csv("results_Kamiran/german_train_repaired.csv", index=False)
    df, _ = dataset_orig_test.convert_to_dataframe(de_dummy_code=True)
    df.to_csv("results_Kamiran/german_test.csv", index=False)

def Kamiran(dataset):
    
    if dataset == 'adult':
        Adult()
    elif dataset == 'compas':
        Compas()
    elif dataset == 'german':
        German()
    