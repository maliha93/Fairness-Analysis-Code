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


def Adult(f):
    X_int = ['age', 'hours_per_week', 'edu_level']
    X_cat = ['marital_status', 'occupation','workclass', 'relationship', 'race', 'native_country']
    S = ['sex']
    Y = ['income']
    df = pd.read_csv(f, header=0, delimiter=',',)
    df = df[X_int+X_cat+S+Y]
    df = df.dropna(how='any', axis=0) 
    train, test = train_test_split(df, test_size=0.3, shuffle=False)
    
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    for i in X_cat:
        df = one_hot(df, i, i) 
    
    X_train, X_test = train_test_split(df, test_size=0.3, shuffle=False)
    train_y = np.array(X_train['income'])
    X_train = X_train.drop('income', axis=1)
    test_y = np.array(X_test['income'])
    X_test = X_test.drop('income', axis=1)
    index = X_train.columns.get_loc('sex')
    clf = LogisticRegression()
    clf.fit(X_train, train_y)
    #y_pred = clf.predict(X_test)
    test['prediction'] = clf.predict_proba(X_test)[:,1]
    test['label'] = test['income']
    test['group'] = X_test['sex']
    train['prediction'] = clf.predict_proba(X_train)[:,1]
    train['label'] = train['income']
    train['group'] = X_train['sex']
    train = train.append(test)
    train.to_csv("data/Adult_post.csv", index=False)
    
    
def Compas(f, f1='', f2=''):
    X_int = ['Prior', 'Age']
    X_cat = ['Sex']
    S = ['Race']
    Y = ['two_year_recid']
    df = pd.read_csv(f, header=0, delimiter=',',)
    df = df[X_int+X_cat+S+Y]
    df = df.dropna(how='any', axis=0) 
    train, test = train_test_split(df, test_size=0.3, shuffle=False)
    
    df['Race'] = df['Race'].map({'Caucasian': 1, 'Hispanic': 1, 'Asian': 1, 'Other': 1, 'Native American': 1, 'African-American': 0})
    for i in X_cat:
        df = one_hot(df, i, i) 
    
    X_train, X_test = train_test_split(df, test_size=0.3, shuffle=False)
    train_y = np.array(X_train['two_year_recid'])
    X_train = X_train.drop('two_year_recid', axis=1)
    test_y = np.array(X_test['two_year_recid'])
    X_test = X_test.drop('two_year_recid', axis=1)
    
    index = X_train.columns.get_loc('Race')
    clf = LogisticRegression()
    clf.fit(X_train, train_y)
    #y_pred = clf.predict(X_test)
    test['prediction'] = clf.predict_proba(X_test)[:,1]
    test['label'] = test['two_year_recid']
    test['group'] = X_test['Race']
    train['prediction'] = clf.predict_proba(X_train)[:,1]
    train['label'] = train['two_year_recid']
    train['group'] = X_train['Race']
    train = train.append(test)
    train.to_csv(f1+"data/Compas_post"+f2+".csv", index=False)
    #metric(index, np.array(X_test), test_y, y_pred)
    
    
def German(f):
    X_int = ['Age','Month','Investment', 'Credit_amount']
    X_cat = ['Status', 'Housing', 'Savings', 'Property', 'Credit_history']
    S = ['Sex']
    Y = ['credit']
    df = pd.read_csv(f, header=0, delimiter=',',)
    df = df[X_int+X_cat+S+Y]
    df = df.dropna(how='any', axis=0) 
    train, test = train_test_split(df, test_size=0.3, shuffle=False)
    
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
    for i in X_cat:
        df = one_hot(df, i, i) 
    
    X_train, X_test = train_test_split(df, test_size=0.3, shuffle=False, random_state=42)
    train_y = np.array(X_train['credit'])
    X_train = X_train.drop('credit', axis=1)
    test_y = np.array(X_test['credit'])
    X_test = X_test.drop('credit', axis=1)
    index = X_train.columns.get_loc('Sex')
    clf = LogisticRegression()
    clf.fit(X_train, train_y)
    #y_pred = clf.predict(X_test)
    #metric(index, np.array(X_test), test_y, y_pred)
    test['prediction'] = clf.predict_proba(X_test)[:,1]
    test['label'] = test['credit']
    test['group'] = X_test['Sex']
    train['prediction'] = clf.predict_proba(X_train)[:,1]
    train['label'] = train['credit']
    train['group'] = X_train['Sex']
    train = train.append(test)
    train.to_csv("data/German_post.csv", index=False)
    
    
    
def make_dataset(dataset, f=None):
    if dataset == 'adult':
        Adult("data/adult.csv")
    elif dataset == 'compas':
        Compas("data/compas.csv")
    elif dataset == 'german':
        German("data/german.csv")
        
