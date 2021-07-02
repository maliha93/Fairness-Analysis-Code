from directindirecteffect import compute_directindirecteffect
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
import time
import sys
import os.path
sys.path.append('../fair_classifiers/')
sys.path.append('../directindirect/')
warnings.filterwarnings('ignore')
from metric import metric, cd


def adult_preprocess(df):
    def income(x):
        if x in ['<=50K', '0', 0, -1]:
            return 0.0
        else:
            return 1.0
        
    def sex(x):
        if x in ['Male', "1", 1]:
            return 1.0
        else:
            return 0.0
        
        
    df['sex'] = df['sex'].apply(lambda x: sex(x))
    df['income'] = df['income'].apply(lambda x: income(x))
    df['pred'] = df['pred'].apply(lambda x: income(x))
    return df

def compas_preprocess(df):
    def two_year_recid(x):
        if x in ['Did recid.', '0', 0, -1]:
            return 0.0
        else:
            return 1.0
        
    def sex(x):
        if x in ['Male', "1", 1]:
            return 1.0
        else:
            return 0.0
        
    def race(x):
        if x in ['African-American']:
            return 0.0
        else:
            return 1.0
        
       
    df['Sex'] = df['Sex'].apply(lambda x: sex(x))
    df['Race'] = df['Race'].apply(lambda x: race(x))
    df['two_year_recid'] = df['two_year_recid'].apply(lambda x: two_year_recid(x))
    df['pred'] = df['pred'].apply(lambda x: two_year_recid(x))
    return df

def german_preprocess(df):
    def credit(x):
        if x in ['Bad Credit', '0', 0, -1]:
            return 0.0
        else:
            return 1.0
        
    def sex(x):
        if x in ['Male', "1", 1]:
            return 1.0
        else:
            return 0.0
        
        
       
    df['Sex'] = df['Sex'].apply(lambda x: sex(x))
    df['credit'] = df['credit'].apply(lambda x: credit(x))
    df['pred'] = df['pred'].apply(lambda x: credit(x))
    return df
    
import sys
sys.path.append('dowhy/')
import numpy as np
import pandas as pd

import dowhy
from dowhy import CausalModel
import dowhy.datasets 

# Avoid printing dataconversion warnings from sklearn
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Config dict to set the logging level
import logging.config
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'WARN',
        },
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)

def adult(df):
    def sex(x):
        if x in ['Male', "1", 1]:
            return True
        else:
            return False
            
    df['sex'] = df['sex'].apply(lambda x: sex(x))
    #print(df)
    df['income'] = df['pred']
    df = df.drop(['pred'], axis=1)
    return df

def compas(df):  
    def race(x):
        if x in ['African-American', 1, "1"]:
            return False
        else:
            return True   
    df['Race'] = df['Race'].apply(lambda x: race(x))
    df['two_year_recid'] = df['pred']
    df = df.drop(['pred'], axis=1)
    return df

def german(df):
    
    def sex(x):
        if x in ['Male', "1", 1]:
            return True
        else:
            return False
    
    df = df.append(df.copy())
    df['Sex'] = df['Sex'].apply(lambda x: sex(x))
    df['credit'] = df['pred']
    df = df.drop(['pred'], axis=1)
    return df


dag = {}
dag['adult'] = 'digraph {race->income; race->occupation; race->hours_per_week; race->edu_level; race->marital_status; age->income; age->occupation; age->hours_per_week; age->workclass; age->marital_status; age->edu_level; age->relationship; sex->occupation; sex->hours_per_week; sex->edu_level; sex->marital_status; sex->relationship; sex->income; native_country->marital_status; native_country->edu_level; native_country->hours_per_week; native_country->workclass; native_country->relationship; native_country->income; marital_status->occupation; marital_status->hours_per_week; marital_status->income; marital_status->workclass; marital_status->edu_level; marital_status->relationship; edu_level->occupation; edu_level->hours_per_week; edu_level->workclass; edu_level->relationship; edu_level->income; occupation->income; hours_per_week->income; workclass->income; relationship->income}'
dag['compas'] = 'digraph {Age->Race; Age->Prior; Age->two_year_recid; Sex->Race; Sex->Prior; Sex->two_year_recid; Race->two_year_recid; Race->Prior; Prior->two_year_recid}'
dag['german'] = 'digraph {Sex->Credit_amount; Sex->Investment; Sex->Savings; Sex->Housing; Sex->Property; Sex->Month; Sex->Status; Sex->Credit_history; Sex->credit; Age->Credit_amount; Age->Investment; Age->Savings; Age->Housing; Age->Property; Age->Month; Age->Status; Age->Credit_history; Credit_amount->credit; Investment->credit; Savings->credit; Housing->credit; Property->credit; Month->credit; Status->credit; Credit_history->credit}'

func = {}
func['adult'] = adult
func['compas'] = compas
func['german'] = german
    


def compute_ate(df, dataset, dataset_info):
    
    t = [dataset_info[dataset]['sens']]
    o = [dataset_info[dataset]['y']]
    g = dag[dataset]
    df = func[dataset](df)
    model=CausalModel(
        data = df,
        treatment=t,
        outcome=o,
        graph=g)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    causal_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_stratification")

    print("Causal Estimate is " + str(causal_estimate.value))
    return causal_estimate.value