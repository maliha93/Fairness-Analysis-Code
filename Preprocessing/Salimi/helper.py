import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
import time
warnings.filterwarnings('ignore')


def one_hot(df, col, pre):
    df_dummy = pd.get_dummies(df[col],prefix=pre,drop_first=True)
    df = pd.concat([df, df_dummy], axis=1)
    df = df.drop(col, axis=1) 
    
    return df

def metric(index, x_test, y_test, y_test_predicted):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i, val in enumerate(x_test):
        if(val[index] == 1):
            if y_test[i]==y_test_predicted[i]==1:
                TP += 1
            if y_test_predicted[i]==1 and y_test[i]!=y_test_predicted[i]:
                FP += 1
            if y_test[i]==y_test_predicted[i]== 0:
                TN += 1
            if y_test_predicted[i]==0 and y_test[i]!=y_test_predicted[i]:
                FN += 1
    TPR_0 = TP/(TP+FN)
    TNR_0 = TN/(FP+TN)
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i, val in enumerate(x_test):
        if(val[index] == 0):
            if y_test[i]==y_test_predicted[i]==1:
                TP += 1
            if y_test_predicted[i]==1 and y_test[i]!=y_test_predicted[i]:
                FP += 1
            if y_test[i]==y_test_predicted[i]==0:
                TN += 1
            if y_test_predicted[i]==0 and y_test[i]!=y_test_predicted[i]:
                FN += 1
    
    TPR = TP/(TP+FN)
    TNR = TN/(FP+TN)
    print("Accuracy:",metrics.accuracy_score(y_test, y_test_predicted))
    print("Precision:",metrics.precision_score(y_test, y_test_predicted))
    print("Recall:",metrics.recall_score(y_test, y_test_predicted))
    print("F1:",metrics.f1_score(y_test, y_test_predicted))
    print("DI: ", di(index, x_test, y_test, y_test_predicted))
    print("TPRB:", TPR_0-TPR)
    print("TNRB:", TNR_0-TNR)

    
def di(index, x_test, y_test, y_pred):
    
    a,b,c,d = 0.0, 0, 0, 0
    for i, val in enumerate(x_test):
        if(val[index] == 0):
            if(y_pred[i] == 1):
                a += 1
            else:
                c += 1
        elif(val[index] == 1):
            if(y_pred[i] == 1):
                b += 1
            else:
                d += 1
    score = (a / (a + c)) / (b / (b + d))
    return score  

import math
def cd(index, x_test, clf):
    
    conf_z = 2.58
    x_test_new = np.zeros(shape=(x_test.shape[0]*2,x_test.shape[1]))
    
    for i, val in enumerate(x_test):
        x_test_new[i*2] = val
        val[index] = (val[index] + 1)%2
        x_test_new[i*2 +1] = val
    
    y_pred = clf.predict(x_test_new)
    count = 0
    for i, val in enumerate(y_pred):
        #print(val)
        if (i%2) == 1:
            continue
        if(val != y_pred[i+1]):
            count = count + 1
            
    cd = (count/x_test.shape[0])
    err = conf_z * math.sqrt((cd * (1 - cd)) / x_test.shape[0])
    print("CD:", cd)
    
    return y_pred


def adult_preprocess(df):
    def income(x):
        if x in ['<=50K', '0', 0]:
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
    return df

def compas_preprocess(df):
    def two_year_recid(x):
        if x in ['Did recid.', '0', 0]:
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
    return df

def german_preprocess(df):
    def credit(x):
        if x in ['Bad Credit', '0', 0]:
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
    return df


def Adult(f):
    X_int = []
    X_cat = ['age', 'edu_level', 'hours_per_week', 'marital_status', 'occupation','workclass', 'relationship',\
             'race', 'native_country']
    S = ['sex']
    Y = ['income']
    keep = ['age', 'edu_level', 'hours_per_week', 'marital_status', 'occupation','workclass', 'relationship',\
            'race', 'native_country', 'sex', 'income']
    
    df = pd.read_csv(f)
    df = df[keep]
    test = pd.read_csv("dataset/adult_bin_test.csv")
    test = test[keep]
    df = pd.concat([df, test])
    df = adult_preprocess(df)
    #df = df.dropna(how='any', axis=0) 
    for i in X_cat:
        if i in keep:
            df = one_hot(df, i, i) 
    
    split = df.shape[0] - test.shape[0]
    X_train, X_test = train_test_split(df, train_size=split, shuffle=False)
    train_y = np.array(X_train['income'])
    X_train = X_train.drop(['income'], axis=1)
    test_y = np.array(X_test['income'])
    X_test = X_test.drop(['income'], axis=1)
    
    clf = LogisticRegression()
    clf.fit(X_train, train_y)
    y_pred = clf.predict(X_test)
    
    index = X_test.columns.get_loc('sex')
    metric(index, np.array(X_test), test_y, y_pred)
    y_cd = cd(index,  np.array(X_test), clf)
    
    test = pd.read_csv("dataset/adult_bin_test.csv")
    test['pred'] = y_pred
    test.to_csv("results_Salimi/adult_test_repaired.csv", index=False)
    np.savetxt("results_Salimi/adult_test_repaired_cd.csv", y_cd, delimiter=",")
    
    
    
def Compas(f, t="dataset/compas_bin_test.csv", f1='', f2=''):
    X_int = []
    X_cat = ['Prior', 'Age', 'Sex']
    keep = ['Prior', 'Age', 'Sex', 'Race', 'two_year_recid']
    S = ['Race']
    Y = ['two_year_recid']
    
    df = pd.read_csv(f)
    df = df[keep]
    test = pd.read_csv(t)
    test = test[keep]
    df = pd.concat([df, test])
    df = compas_preprocess(df)
    for i in X_cat:
        if i in keep:
            df = one_hot(df, i, i) 
    
    split = df.shape[0] - test.shape[0]
    X_train, X_test = train_test_split(df, train_size=split, shuffle=False)
    train_s = np.array(X_train['Race'])
    train_y = np.array(X_train['two_year_recid'])
    X_train = X_train.drop(['two_year_recid'], axis=1)
    test_y = np.array(X_test['two_year_recid'])
    test_s = np.array(X_test['Race'])
    X_test = X_test.drop(['two_year_recid'], axis=1)
    
    clf = LogisticRegression()
    clf.fit(X_train, train_y)
    y_pred = clf.predict(X_test)
    
    #X_test['Race'] = test_s
    index = X_test.columns.get_loc('Race')
    metric(index, np.array(X_test), test_y, y_pred)
    y_cd = cd(index,  np.array(X_test), clf)
    
    
    test = pd.read_csv(t)
    test['pred'] = y_pred
    test.to_csv(f1+"results_Salimi/compas_test_repaired"+f2+".csv", index=False)
    np.savetxt(f1+"results_Salimi/compas_test_repaired"+f2+"_cd.csv", y_cd, delimiter=",")
    
    
    
def German(f):
    X_int = []
    X_cat = ['Age', 'Month', 'Investment', 'Status', 'Housing', 'Savings', 'Property', 'Credit_history']
    S = ['Sex']
    Y = ['credit']
    keep = X_int+X_cat+S+Y
    
    df = pd.read_csv(f)
    df = df[keep]
    test = pd.read_csv("dataset/german_bin_test.csv")
    test = test[keep]
    df = pd.concat([df, test])
    df = german_preprocess(df)
    
    for i in X_cat:
        if i in keep:
            df = one_hot(df, i, i) 
    
    split = df.shape[0] - test.shape[0]
    X_train, X_test = train_test_split(df, train_size=split, shuffle=False)
    train_s = np.array(X_train['Sex'])
    train_y = np.array(X_train['credit'])
    X_train = X_train.drop(['credit'], axis=1)
    test_y = np.array(X_test['credit'])
    test_s = np.array(X_test['Sex'])
    X_test = X_test.drop(['credit'], axis=1)
    
    
    clf = LogisticRegression()
    clf.fit(X_train, train_y)
    y_pred = clf.predict(X_test)
    
    #X_test['Sex'] = test_s
    index = X_test.columns.get_loc('Sex')
    metric(index, np.array(X_test), test_y, y_pred)
    y_cd = cd(index,  np.array(X_test), clf)
    
    
    test = pd.read_csv("dataset/german_bin_test.csv")
    test['pred'] = y_pred
    test.to_csv("results_Salimi/german_test_repaired.csv", index=False)
    np.savetxt("results_Salimi/german_test_repaired_cd.csv", y_cd, delimiter=",")
    
    
    
    
def compute_metrics(dataset, f=None):
    if dataset == 'adult':
        Adult(f)
    elif dataset == 'compas':
        Compas(f)
    elif dataset == 'german':
        German(f)
    elif dataset == 'credit':
        Credit(f)
        
#compute_metrics('german', "results_German/train__repMF_0.csv")