from sklearn import metrics

def SPD(sens, y_test, y_pred):
    
    priv_0, priv_1, prot_0, prot_1 = 0, 0, 0, 0
    for i, val in enumerate(sens):
        if(val == 0):
            if(y_pred[i] == 1):
                prot_1 += 1
            else:
                prot_0 += 1
        elif(val == 1):
            if(y_pred[i] == 1):
                priv_1 += 1
            else:
                priv_0 += 1
    
    score = (priv_1 / (priv_1 + priv_0)) - (prot_1 / (prot_0 + prot_1))
    print("SPD:", score)
    return score 

def correctness(y_test, y_test_predicted):
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_test_predicted))
    print("Precision:",metrics.precision_score(y_test, y_test_predicted))
    print("Recall:",metrics.recall_score(y_test, y_test_predicted))
    
def EO_PP(sens, y_test, y_test_predicted):
    
    TP, FP, TN, FN = 0, 0, 0, 0
    for i, val in enumerate(sens):
        if(val == 1):
            if y_test[i]==y_test_predicted[i]==1:
                TP += 1
            if y_test_predicted[i]==1 and y_test[i]!=y_test_predicted[i]:
                FP += 1
            if y_test[i]==y_test_predicted[i]== -1:
                TN += 1
            if y_test_predicted[i]==-1 and y_test[i]!=y_test_predicted[i]:
                FN += 1
    TPR_1 = TP/(TP+FN)
    TNR_1 = TN/(FP+TN)
    
    
    TP, FP, TN, FN = 0, 0, 0, 0   
    for i, val in enumerate(sens):
        if(val == 0):
            if y_test[i]==y_test_predicted[i]==1:
                TP += 1
            if y_test_predicted[i]==1 and y_test[i]!=y_test_predicted[i]:
                FP += 1
            if y_test[i]==y_test_predicted[i]==-1:
                TN += 1
            if y_test_predicted[i]==-1 and y_test[i]!=y_test_predicted[i]:
                FN += 1   
    TPR_0 = TP/(TP+FN)
    TNR_0 = TN/(FP+TN)
    print("EO:", TPR_1-TPR_0, "PP:", TNR_1-TNR_0)
    
    return TPR_1-TPR_0, TNR_1-TNR_0