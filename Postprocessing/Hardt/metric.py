from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np

def metric(priv_label, priv_pred, prot_label, prot_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    a, b, c, d = 0, 0 , 0, 0
    for i, val in enumerate(priv_label):
        
        if(priv_pred[i].round()==0):
            b += 1
        else:
            d += 1
          
        if priv_label[i]==priv_pred[i].round()==1:
            TP += 1
        if priv_pred[i].round()==1 and priv_label[i]!=priv_pred[i].round():
            FP += 1
        if priv_label[i]==priv_pred[i].round()== 0:
            TN += 1
        if priv_pred[i].round()==0 and priv_label[i]!=priv_pred[i].round():
            FN += 1
    TPR1 = TP/(TP+FN)
    TNR1 = TN/(FP+TN)
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i, val in enumerate(prot_label):
        
        if(prot_pred[i].round()==0):
            a += 1
        else:
            c += 1
        if prot_label[i]==prot_pred[i].round()==1:
            TP += 1
        if prot_pred[i].round()==1 and prot_label[i]!=prot_pred[i].round():
            FP += 1
        if prot_label[i]==prot_pred[i].round()== 0:
            TN += 1
        if prot_pred[i].round()==0 and prot_label[i]!=prot_pred[i].round():
            FN += 1
    TPR = TP/(TP+FN)
    TNR = TN/(FP+TN)
    
    correctness(priv_label, priv_pred, prot_label, prot_pred)
    score = (a * (b + d)) / ((a + c) * b + 0.00001)
    print("DI: ", score)
    print("TPRB:", TPR1-TPR)
    print("TNRB:", TNR1-TNR)

import math    
def cd(x1, x2, y1, y2):
    conf_z = 2.58
    x = np.append(x1, x2)
    y = np.append(y1, y2)
    res = []
    count = 0
    for i, val in enumerate(x):
        if val != y[i]:
            count = count+1
        res.append(val)
        res.append(y[i])
            
    cd = (count/x.shape[0])
    err = conf_z * math.sqrt((cd * (1 - cd)) / x.shape[0])
    print("CD:", cd)
    return np.array(res)
    
def correctness(priv_label, priv_pred, prot_label, prot_pred):
    
    y_true = np.append(priv_label, prot_label)
    y_pred = np.append(priv_pred, prot_pred)
    y_true = np.around(y_true)
    y_pred = np.around(y_pred)
    print("Accuracy:",metrics.accuracy_score(y_true, y_pred))
    print("Precision:",metrics.precision_score(y_true, y_pred))
    print("Recall:",metrics.recall_score(y_true, y_pred))
    print("Recall:",metrics.f1_score(y_true, y_pred))