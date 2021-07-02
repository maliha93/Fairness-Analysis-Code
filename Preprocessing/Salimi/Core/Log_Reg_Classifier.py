import numpy as np
import numpy as np
import pandas as pd
from sklearn import preprocessing


import pprint
from os import chdir
from sklearn.ensemble import RandomForestClassifier
import sys
#sys.path.insert(0, '//Users/babakmac/Documents/HypDB/relational-causal-inference/source/HypDB')
#from core.cov_selection import *
#from core.explanation import *
#import core.query as sql
#import modules.statistics.cit as ci_test
#from Modules.InformationTheory.info_theo import *
from sklearn.metrics import confusion_matrix
import copy
from sklearn import tree
from utils.read_data import read_from_csv
from sklearn import model_selection
from sklearn.model_selection import cross_val_score


import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
from scipy import interp
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import math
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
def data_split(data,outcome,path,k=5,test_size=0.3):


    rs = StratifiedShuffleSplit(n_splits=k, test_size=test_size, random_state=2)
    data_y = pd.DataFrame(data[outcome])
    data_X = data.drop([outcome], axis=1)
    rs.get_n_splits(data_X, data_y)
    j = 0
    for test, train in rs.split(data_X,data_y):
        cur_test  = data.iloc[train]
        cur_train = data.iloc[test]
        cur_train = pd.concat([cur_test, cur_train])
        cur_train.to_csv(path + 'train_' + str(j) + '.csv', encoding='utf-8', index=False)
        #print(path + 'train_' + str(j) + '.csv')
        #cur_test.to_csv(path + 'test_' + str(j) + '.csv', encoding='utf-8', index=False)
        #print(len(cur_test.index))
        #print(path + 'test_' + str(j) + '.csv')
        j +=1

def cross_valid(data,features,D_features,Y_features,X_features,path,k=5):

    print('Original Data Size',len(data.index))
    train_df = data[features]
    dft1 = pd.get_dummies(train_df[X_features])
    dft2 = pd.get_dummies(train_df[Y_features])

    X = dft1.values
    y = dft2.values
    y = y.flatten()
    cv = StratifiedKFold(n_splits=k,shuffle=True)
    #classifier = LogisticRegression()
    j = 0
    for train, test in cv.split(X, y):
        cur_train = train_df.iloc[train]
        cur_test = train_df.iloc[test]
        cur_train.to_csv(path + 'train_' + str(j) + '.csv', encoding='utf-8', index=False)
        print(len(cur_train.index))
        print(path + 'train_' + str(j) + '.csv')
        cur_test.to_csv(path + 'test_' + str(j) + '.csv', encoding='utf-8', index=False)
        print(len(cur_test.index))
        print(path + 'test_' + str(j) + '.csv')
        j +=1

def strr(list):
   return str(['%.3f' % val for val in  list])


def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('*****************************************************************************************')
         print('\t' * (indent+1) + strr(value))
         print('mean:', mean(value))
         print('variance:', var(value))
         print('*****************************************************************************************')


def test_rep_str(D_features,Y_features,X_features,path1,path2,k=5,droped=False,classifier='log_reg'):
    if classifier=='log_reg':
        classifier = LogisticRegression()
    elif classifier=='rand_forest':
        classifier=RandomForestClassifier(max_depth=2, random_state=0)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    MI_inp = dict()
    MI_out = dict()
    MI_test=dict()
    for j in range(0, k):
        print(path2+str(j)+'.csv')
        cur_train=read_from_csv(path1+str(j)+'.csv')
        print(path1+str(j)+'.csv')
        cur_test=read_from_csv(path2+str(j)+'.csv')
        #atts=cur_train.columns
        #atts=atts.tolist()
        #list=[att.replace('_x','').replace('_y','') for att in atts]
        #atts


        for item in D_features:

            pval, mi = ci_test.ulti_fast_permutation_tst(cur_train, item, Y_features, X_features, pvalue=0.01,
                                                        debug=False, loc_num_samples=100,
                                                        num_samples=100, view=False)
            rmi = round(mi, 3)
            print('####################################')
            print(len(cur_train.index))
            print('Mutul information in train data:', item,'pvalue:' , pval, 'MI:', rmi)
            print('####################################')
            if  item not in MI_inp.keys():
                MI_inp[item]= [rmi]
            else:
                MI_inp[item] = MI_inp[item] +[rmi]



        inf = Info(cur_test)

        for item in D_features:

            pval, mi = ci_test.ulti_fast_permutation_tst(cur_test, item, Y_features, X_features, pvalue=0.01,
                                                        debug=False, loc_num_samples=100,
                                                        num_samples=100, view=False)
            mi = round(mi, 3)
            print('####################################')
            print('MI in test data:', item,'pvalue:' , pval, 'MI:', mi)
            print('####################################')
            if  item not in MI_test.keys():
                MI_test[item]= [mi]
            else:
                MI_test[item] = MI_test[item] +[mi]




        mi = inf.CMI(D_features+X_features, Y_features)
        mi = round(mi, 3)
        print('Predictive Power(traning)', mi)


        inf = Info(cur_test)
        mi = inf.CMI(D_features, Y_features,X_features)
        mi = round(mi, 3)
        print('Repaied MI test', mi)

        mi = inf.CMI(D_features+X_features, Y_features)
        mi = round(mi, 3)
        print('Predictive Power(test)', mi)

        cur_train[Y_features[0]] = pd.to_numeric(cur_train[Y_features[0]])
        ate = cur_train.groupby([D_features[0]])[Y_features[0]].mean()
        print(ate)
#        m = abs(ate.values[0] - ate.values[1]).value
        #ate0.insert(0, m)
        #print('Repaied ATE \n', ate)
#        new=abs(max((ate.values[0] / ate.values[1]) - 1, (ate.values[0] / ate.values[1]) - 1)).value
        #print('Repaied J \n', new)
        #J1.insert(0,new)


        #ate = cur_test.groupby([D_features[0]])[Y_features[0]].mean()
        #m = abs(ate.values[0] - ate.values[1]).value
        #ate0.insert(0, m)
        #print('Repaied ATE test \n', ate)
        #new=abs(max((ate.values[0] / ate.values[1]) - 1, (ate.values[0] / ate.values[1]) - 1)).value
        #print('Repaied J test \n', new)
        #J1.insert(0,new)

        # print("len",cur_train.columns,len(cur_train.index),cur_train.shape)
        # print("len",len(cur_test.index),cur_test.shape)

        j += 1

        #inf = Info(cur_train)

        #MI_inp.insert(0, I)
        cur_test['W']=1
        train_objs_num = len(cur_train)
        dataset = pd.concat(objs=[cur_train[ D_features+X_features], cur_test[ D_features+X_features]], axis=0)
        dataset = pd.get_dummies(dataset)
        dft1 = dataset[:train_objs_num]
        dft4 = dataset[train_objs_num:]

        train_X = dft1.values
        train_y = cur_train[Y_features[0]].values
        # train_y=train_y.flatten()


        #if droped:
        #    dft4 = pd.get_dummies(cur_test[X_features])
        #else:
        #    dft4 = pd.get_dummies(cur_test[ D_features+X_features])
        #print(cur_test[D_features+X_features])






        dft5 = pd.get_dummies(cur_test[Y_features])

        # logit = sm.Logit(train_df['bscore'], train_df['juv_misd_count'])
        X = dft4.values
        y = dft5.values
        y = y.flatten()

        #print("#####################",len(train_X),len(train_y),type(train_X),type(train_y),train_X,train_y,X.shape)
        print(X.shape,train_X.shape)

        kfold = model_selection.KFold(n_splits=10, random_state=7)
        modelCV = LogisticRegression()

        probas_ = classifier.fit(train_X, train_y).predict_proba(X)

        scoring = 'accuracy'

        results = model_selection.cross_val_score(modelCV, train_X, train_y, cv=kfold, scoring=scoring)

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',mean(results))
        #logit = sm.Logit(train_X,cur_train[Y_features[0]])

        # fit the model
        #result = logit.fit()


        #print(probas_)


        y_pred = classifier.predict(X)

        cur_test.insert(0,'y',y_pred)    # insert the outcome into the test dataset

        for item in D_features:
            pval, mi = ci_test.ulti_fast_permutation_tst(cur_test, item, ['y'], X_features, pvalue=0.01,
                                                         debug=False, loc_num_samples=100,
                                                         num_samples=100, view=False)
            mi = round(mi, 3)
            print('*************************')
            print(' MI in output',item,'pvalue:' , pval, 'MI:', mi)
            print('***************************')
            if item not in MI_out.keys():
                MI_out[item] = [mi]
            else:
                MI_out[item] = MI_out[item] + [mi]
        print(path1 + str(j) + '.csv')


        for item in D_features:
            pval, mi = ci_test.ulti_fast_permutation_tst(cur_test, item, ['y'], pvalue=0.01,
                                                         debug=False, loc_num_samples=100,
                                                         num_samples=100, view=False)
            #mi = round(mi, 3)
            print('*************************')
            print(' MI in output (marginal)',item,'pvalue:' , pval, 'MI:', mi)
            print('***************************')


        ate = cur_test.groupby([D_features[0]])[['y']].mean()
        print(ate)
        # print("ATE on on test labels", '\n averagee:', mean(ate1), "variancee", var(ate1))
        # print("ATE on on outcome", '\n averagee:', mean(ate2), "variancee", var(ate2))
        # print("J on on input", '\n averagee:', mean(J1), "variancee", var(J1))
        # print("J on on outcome", '\n averagee:', mean(J2), "variancee", var(J2))
        print('####################################')


        #ate = cur_test.groupby(D_features)[Y_features[0]].mean()
        #m = abs(ate.values[0] - ate.values[1]).value
        #ate1.insert(0, m)

        ate = cur_test.groupby(D_features)['y'].mean()
        #m = abs(ate.values[0] - ate.values[1]).value
        #ate2.insert(0, m)
        print('ATE on outcome:',ate)
        #new=abs(max((ate.values[0] / ate.values[1]) - 1, (ate.values[0] / ate.values[1]) - 1)).value
        #print('Outcome J \n', new)
        #J2.insert(0,new)

        fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        cur_test.to_csv(path1 + '_trained.csv')
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #print("Mutual Information on repaired traning labels", '\n averagee:', mean(rep_MI_inp), "variancee",var(rep_MI_inp))
    #print("ATE on repaired traning labels", '\n averagee:', mean(ate0), "variancee", var(ate0))
    #print("Mutual Information on test labels", '\n averagee:', mean(MI_inp.values()), "variancee", var(MI_inp.values()))
    #print("Mutual Information on outcome", '\n avg:', mean(MI_out.values()), "variancee", var(MI_out.values()))

    print("Mutual Information on train: \n")
    pretty(MI_inp)
    plt.show()
    print("Mutual Information on test: \n")
    pretty(MI_test)
    #print(" Mutual Information on repaired data",   rep_MI_inp)
    print("Mutual Information on outcome: \n")
    pretty(MI_out)
    plt.show()
    return MI_out,MI_inp, mean_auc, std_auc



def classification(cur_train,cur_test, dependant, dependee, classifier='log_reg'):

    if classifier=='log_reg':
        classifier = LogisticRegression()
    elif classifier=='rand_forest':
        classifier=RandomForestClassifier(max_depth=2, random_state=0)
    train_objs_num = len(cur_train)
    dataset = pd.concat(objs=[cur_train[dependant], cur_test[ dependant]], axis=0)
    dataset = pd.get_dummies(dataset)
    dft1 = dataset[:train_objs_num]
    dft4 = dataset[train_objs_num:]
    train_X = dft1.values
    train_y = cur_train[dependee[0]].values
    dft5 = pd.get_dummies(cur_test[dependee])
    X = dft4.values
    y = dft5.values
    y = y.flatten()
    probas_ = classifier.fit(train_X, train_y).predict_proba(X)
    #coef=  classifier.coef_
    y_pred = classifier.predict(X)
    probas_=np.array(probas_)
    #cur_test.insert(0, 'prob', probas_[:,0])
    cur_test.insert(0,'y',y_pred)    # insert the outcome into the test dataset
        #cur_test['FP']=cur_test.loc[(cur_test[Y_features] ==1) & (cur_test.y == 1)]
    #cur_test['FP'] = cur_test.apply(lambda x: 1 if x[dependee[0]] == 0 and x['y'] == 1 else 0, axis=1)
    #cur_test['FN'] = cur_test.apply(lambda x: 1 if x[dependee[0]] == 1 and x['y'] == 0 else 0, axis=1)
    print('accuracy',accuracy_score(cur_test[dependee[0]], y_pred, normalize=True))
    print('AUC', roc_auc_score(cur_test[dependee[0]], y_pred))
    print(confusion_matrix(cur_test[dependee[0]], y_pred))

    fpr, tpr, _ = roc_curve(y_pred, cur_test[dependee[0]], drop_intermediate=False)

    import matplotlib.pyplot as plt
    plt.figure()
    ##Adding the ROC
    plt.plot(fpr, tpr, color='red',
             lw=2, label='ROC curve')
    ##Random FPR and TPR
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    ##Title and label
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.show()
    return cur_test



def old_test_rep_str(indeps, features, protecteds, Y_features, path1, path2, k=5, droped=False, classifier='log_reg',method='original'):
    classifer_method=classifier
    if Y_features[0] in features:
        features.remove(Y_features[0])


    D_features=[]
    X_features=features
    print("Fetures to learn on",X_features+D_features)
    if classifier=='log_reg':
        classifier = LogisticRegression()
    elif classifier=='rand_forest':
        classifier=RandomForestClassifier(max_depth=2, random_state=0)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for j in range(0, k):
        print(path2+str(j)+'.csv')
        cur_train=read_from_csv(path1+str(j)+'.csv')
        print(path1+str(j)+'.csv')
        cur_test=read_from_csv(path2+str(j)+'.csv')
        #atts=cur_train.columns
        #atts=atts.tolist()
        #list=[att.replace('_x','').replace('_y','') for att in atts]
        #atts
        for att in protecteds:
            ate = cur_train.groupby([att])[Y_features].mean()
            print('ATE on train:',att, ate)

        for att in protecteds:
            ate = cur_test.groupby([att])[Y_features].mean()
            print('ATE on test:',att, ate)

        i=0
        for indep in indeps:
                X=indep[0]
                Y=indep[1]
                Z=indep[2]
                for att in [X,Y,Z]:
                    if 'y' in att:
                        att.remove('y')
                        att.insert(0,Y_features[0])
                pval, mi = ci_test.ulti_fast_permutation_tst(cur_train, X, Y, Z, pvalue=0.01,
                                                            debug=False, loc_num_samples=100,
                                                            num_samples=100, view=False)
                rmi = round(mi, 3)
                print('####################################')
                print(len(cur_train.index))
                print('MI in train data:', indep,'pvalue:' , pval, 'MI:', rmi)
                print('####################################')
                MI_inp[i]= [rmi]
                i+=1



        inf = Info(cur_test)

        i=0
        for indep in indeps:
                X=indep[0]
                Y=indep[1]
                Z=indep[2]
                for att in [X,Y,Z]:
                    if 'y' in att:
                        att.remove('y')
                        att.insert(0,Y_features[0])
                pval, mi = ci_test.ulti_fast_permutation_tst(cur_test, X, Y, Z, pvalue=0.01,
                                                            debug=False, loc_num_samples=100,
                                                            num_samples=100, view=False)
                rmi = round(mi, 3)
                print('####################################')
                print(len(cur_test.index))
                print('MI in test data:', indep,'pvalue:' , pval, 'MI:', rmi)
                print('####################################')
                MI_test[i]= [rmi]
                i+=1




        mi = inf.CMI(D_features+X_features, Y_features)
        mi = round(mi, 3)
        print('Predictive Power(traning)', mi)


        inf = Info(cur_test)
        mi = inf.CMI(D_features, Y_features,X_features)
        mi = round(mi, 3)
        print('Repaied MI test', mi)

        mi = inf.CMI(D_features+X_features, Y_features)
        mi = round(mi, 3)
        print('Predictive Power(test)', mi)

        #cur_train[Y_features[0]] = pd.to_numeric(cur_train[Y_features[0]])
        #ate = cur_train.groupby([D_features[0]])[Y_features[0]].mean()
        #print(ate)
#        m = abs(ate.values[0] - ate.values[1]).value
        #ate0.insert(0, m)
        #print('Repaied ATE \n', ate)
#        new=abs(max((ate.values[0] / ate.values[1]) - 1, (ate.values[0] / ate.values[1]) - 1)).value
        #print('Repaied J \n', new)
        #J1.insert(0,new)


        #ate = cur_test.groupby([D_features[0]])[Y_features[0]].mean()
        #m = abs(ate.values[0] - ate.values[1]).value
        #ate0.insert(0, m)
        #print('Repaied ATE test \n', ate)
        #new=abs(max((ate.values[0] / ate.values[1]) - 1, (ate.values[0] / ate.values[1]) - 1)).value
        #print('Repaied J test \n', new)
        #J1.insert(0,new)

        # print("len",cur_train.columns,len(cur_train.index),cur_train.shape)
        # print("len",len(cur_test.index),cur_test.shape)

        j += 1

        #inf = Info(cur_train)

        #MI_inp.insert(0, I)
        cur_test['W']=1
        train_objs_num = len(cur_train)
        dataset = pd.concat(objs=[cur_train[ D_features+X_features], cur_test[ D_features+X_features]], axis=0)
        dataset = pd.get_dummies(dataset)
        dft1 = dataset[:train_objs_num]
        dft4 = dataset[train_objs_num:]

        train_X = dft1.values
        train_y = cur_train[Y_features[0]].values
        # train_y=train_y.flatten()


        #if droped:
        #    dft4 = pd.get_dummies(cur_test[X_features])
        #else:
        #    dft4 = pd.get_dummies(cur_test[ D_features+X_features])
        #print(cur_test[D_features+X_features])






        dft5 = pd.get_dummies(cur_test[Y_features])

        # logit = sm.Logit(train_df['bscore'], train_df['juv_misd_count'])
        X = dft4.values
        y = dft5.values
        y = y.flatten()

        #print("#####################",len(train_X),len(train_y),type(train_X),type(train_y),train_X,train_y,X.shape)
        print(X.shape,train_X.shape)
        #logit = sm.Logit(train_X,cur_train[Y_features[0]])


        probas_ = classifier.fit(train_X, train_y).predict_proba(X)

        scoring = 'accuracy'
        #result = logit.fit()


        #print(probas_)
        y_pred = classifier.predict(X)
        #predicted = cross_validation.cross_val_predict(logreg, X, y, cv=10)

        cur_test.insert(0,'y',y_pred)    # insert the outcome into the test dataset
        #cur_test['FP']=cur_test.loc[(cur_test[Y_features] ==1) & (cur_test.y == 1)]
        cur_test['FP'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 0 and x['y'] == 1 else 0, axis=1)
        cur_test['FN'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 1 and x['y'] == 0 else 0, axis=1)
        cur_test['TP'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 1 and x['y'] == 1 else 0, axis=1)
        cur_test['TN'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 0 and x['y'] == 0 else 0, axis=1)

        i=0
        Z=[]
        #t_indeps=indeps.copy()
        t_indeps=copy.copy(indeps)
        for indep in t_indeps:
                X=indep[0]
                Y=indep[1]
                Z=indep[2]
                for att in [X,Y,Z]:
                    if Y_features[0] in att:
                        att.remove(Y_features[0])
                        att.insert(0,'y')
                for p in protecteds:
                        if p in Z:
                            Z.remove(p)
                pval, mi = ci_test.ulti_fast_permutation_tst(cur_test, X, Y, Z, pvalue=0.01,
                                                            debug=False, loc_num_samples=100,
                                                            num_samples=100, view=False)
                rmi = round(mi, 3)
                print('####################################')
                print(len(cur_test.index))
                print('MI in outcome:', indep,'pvalue:' , pval, 'MI:', rmi)
                print('####################################')
                MI_test[i]= [rmi]
                i+=1


        print(path1 + str(j) + '.csv')


        for item in D_features:
            pval, mi = ci_test.ulti_fast_permutation_tst(cur_test, item, ['y'], pvalue=0.01,
                                                         debug=False, loc_num_samples=100,
                                                         num_samples=100, view=False)
            #mi = round(mi, 3)
            print('*************************')
            print(' MI in output (marginal)',item,'pvalue:' , pval, 'MI:', mi)
            print('***************************')

        inf=Info(cur_test)
        for att in protecteds:
            ate = cur_test.groupby([att])[['y']].mean()
            print(att, ate)
            #sql.mplot(ate, att, ['y'], 'Average Outcome', att,
            #         fontsize=20)

            ate, matcheddata, adj_set, pur = sql.adjusted_groupby(cur_test, [att], ['y'],
                                                                  threshould=10,
                                                                  covariates=Z, mediatpor=[], init=['Male'])

            print('adjusted', att, X_features, ate)
            #sql.mplot(ate, [att], ['y'], '  Average Outcome Adjusted by:  ' + list2string(Z),
            #         list2string([att]),
            #         fontsize=24)

            print('*************************')
            ate = cur_test.groupby([att])[['FP']].mean()
            print('FP:', att, ate)
            #sql.mplot(ate, att, ['FP'], 'FP Rate', att, fontsize=24)
            ate = cur_test.groupby([att])[['FN']].mean()
            print('FN:', att, ate)
            #sql.mplot(ate, [att], ['FN'], 'FN Rate', att, fontsize=24)
            print('***************************')

            ate, matcheddata, adj_set, pur = sql.adjusted_groupby(cur_test, [att], ['FP'],
                                                                  threshould=0,
                                                                  covariates=Z, mediatpor=[], init=['Male'])
            #sql.mplot(ate, [att], ['FP'], 'FP Rate Adjusted by:  '  + list2string(Z),
            #         list2string([att]),
            #         fontsize=24)
            print('Adjusted FP:', att, inf.CMI([att], ['FP']), inf.CMI([att], ['FP'], Z))
            print('***************************')

            ate, matcheddata, adj_set, pur = sql.adjusted_groupby(cur_test, [att], ['FN'],
                                                                  threshould=10,
                                                                  covariates=Z, mediatpor=[], init=['Male'])
            #sql.mplot(ate, [att], ['FN'], 'FN Rate Adjusted by:  '  + list2string(Z),
            #         list2string([att]),
            #         fontsize=24)
            print('Adjusted FN:', att, ate, inf.CMI([att], ['FN']), inf.CMI([att], ['FN'], Z))
            print('***************************')



            '####'

        #print('Adjusted FP:', att, inf.CMI([att], ['FP']), inf.CMI([att], ['FP'], Z+Y_features))
        #print('***************************')

        #ate, matcheddata, adj_set, pur = sql.adjusted_groupby(cur_test, [att], ['FN'],
        #                                                      threshould=10,
        #                                                      covariates=Z, mediatpor=[], init=['Male'])
        #sql.mplot(ate, [att], ['FN'], 'FN Rate Adjusted by:  ' + list2string(Z),
        #          list2string([att]),
        #          fontsize=24)
        #print('Adjusted FN:', att, ate, inf.CMI([att], ['FN']), inf.CMI([att], ['FN'], Z+Y_features))
        print('***************************')



        #ate = cur_test.groupby(D_features)[Y_features[0]].mean()
        #m = abs(ate.values[0] - ate.values[1]).value
        #ate1.insert(0, m)

        #ate = cur_test.groupby(D_features)['y'].mean()
        #m = abs(ate.values[0] - ate.values[1]).value
        #ate2.insert(0, m)
        #new=abs(max((ate.values[0] / ate.values[1]) - 1, (ate.values[0] / ate.values[1]) - 1)).value
        #print('Outcome J \n', new)
        #J2.insert(0,new)

        fpr, tpr, thresholds = roc_curve(cur_test[Y_features[0]], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        #df=cur_test[features+['FP','FN','y']]
        df=cur_test
        ##FPs1=df[df['y']==1].index
        #FPs2 = df[df[Y_features] == 0].index
        #index=intersect1d(FPs1,FPs2)
        #df[FP]=
        #M=confusion_matrix(cur_test['y'], cur_test[Y_features])
        #print(M)
        df.to_csv(path1 + '_'+method+'_'+classifer_method+'_'+str(j)+'.csv')
        print(path1 + '_'+method+'_'+classifer_method+'_'+str(j)+'.csv')
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    #print("Mutual Information on repaired traning labels", '\n averagee:', mean(rep_MI_inp), "variancee",var(rep_MI_inp))
    #print("ATE on repaired traning labels", '\n averagee:', mean(ate0), "variancee", var(ate0))
    #print("Mutual Information on test labels", '\n averagee:', mean(MI_inp.values()), "variancee", var(MI_inp.values()))
    #print("Mutual Information on outcome", '\n avg:', mean(MI_out.values()), "variancee", var(MI_out.values()))

    print("MI on train: \n")
    pretty(MI_inp)
    plt.show()
    print("MI on test: \n")
    pretty(MI_test)
    #print(" Mutual Information on repaired data",   rep_MI_inp)
    print("MI on outcome: \n")
    pretty(MI_out)
    plt.show()
    #print('Coeffient', coef)
    return aucs


def new_test_rep_str(indeps, features, protecteds, Y_features, path1, path2, k=5, droped=False, classifier='log_reg',method='original'):
    clf=classifier
    if Y_features[0] in features:
        features.remove(Y_features[0])

    D_features=[]
    X_features=features
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for j in range(0, k):
        print(path2+str(j)+'.csv')
        cur_train=read_from_csv(path1+str(j)+'.csv')
        print(path1+str(j)+'.csv')
        cur_test=read_from_csv(path2+str(j)+'.csv')
        #atts=cur_train.columns
        #atts=atts.tolist()
        #list=[att.replace('_x','').replace('_y','') for att in atts]
        #atts
        train_objs_num = len(cur_train)
        dataset = pd.concat(objs=[cur_train[ D_features+X_features], cur_test[ D_features+X_features]], axis=0)
        dataset = pd.get_dummies(dataset)
        dft1 = dataset[:train_objs_num]
        dft4 = dataset[train_objs_num:]

        train_X = dft1.values
        train_y = cur_train[Y_features[0]].values
        # train_y=train_y.flatten()

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        #if droped:
        #    dft4 = pd.get_dummies(cur_test[X_features])
        #else:
        #    dft4 = pd.get_dummies(cur_test[ D_features+X_features])
        #print(cur_test[D_features+X_features])






        dft5 = pd.get_dummies(cur_test[Y_features])

        # logit = sm.Logit(train_df['bscore'], train_df['juv_misd_count'])
        X = dft4.values
        y = dft5.values
        y = y.flatten()

        #print("#####################",len(train_X),len(train_y),type(train_X),type(train_y),train_X,train_y,X.shape)
        print(X.shape,train_X.shape)
        #logit = sm.Logit(train_X,cur_train[Y_features[0]])


        probas_ = classifier.fit(train_X, train_y).predict_proba(X)

        scoring = 'accuracy'
        #result = logit.fit()


        #print(probas_)
        y_pred = classifier.predict(X)
        #predicted = cross_validation.cross_val_predict(logreg, X, y, cv=10)

        cur_test.insert(0,'y',y_pred)    # insert the outcome into the test dataset
        #cur_test['FP']=cur_test.loc[(cur_test[Y_features] ==1) & (cur_test.y == 1)]
        cur_test['FP'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 0 and x['y'] == 1 else 0, axis=1)
        cur_test['FN'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 1 and x['y'] == 0 else 0, axis=1)
        cur_test['TP'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 1 and x['y'] == 1 else 0, axis=1)
        cur_test['TN'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 0 and x['y'] == 0 else 0, axis=1)

        i=0
        Z=[]
        fpr, tpr, thresholds = roc_curve(cur_test[Y_features[0]], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        #df=cur_test[features+['FP','FN','y']]
        df=cur_test
        ##FPs1=df[df['y']==1].index
        #FPs2 = df[df[Y_features] == 0].index
        #index=intersect1d(FPs1,FPs2)
        #df[FP]=
        #M=confusion_matrix(cur_test['y'], cur_test[Y_features])
        #print(M)
        df.to_csv(path1 + '_'+method+'_'+clf.__class__.__name__+'_'+str(j)+'.csv')
        print(path1 + '_'+method+'_'+clf.__class__.__name__+'_'+str(j)+'.csv')
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return aucs

if __name__ == "__main__":
    #train_df = pd.read_csv("/Users/babakmac/Documents/XDBData/ad_data_prep.csv")
    #train_df = pd.read_csv("../ data / rep_test_df.csv")
    train_df = pd.read_csv("/Users/babakmac/Documents/XDBData/cmu_compas.csv")
    #train_df = pd.read_csv("../data/rep_test_df.csv")
    #features = ['race', 'age_cat', 'c_charge_degree', 'priors_count', 'is_recid']
    features = ['education', 'occupation', 'age', 'race', 'sex', 'income', 'maritalstatus']
    D2_features = ['race', 'sex', 'age']
    Y2_features = ['income', 'maritalstatus']
    X2_features = ['hoursperweek', 'education', 'occupation']

    D_features = ['race']
    Y_features = ['income']
    X_features = ['hoursperweek', 'education', 'occupation','sex', 'age','maritalstatus']

    #cross_valid(train_df, features, D_features, Y_features, X_features, path, k=10)
    method = 'sat'
    smother=1
    size = 5000000
    smothers = [1]
    folds=5

    #path1 = path + 'train__rep' + method + '_' + str(size) + '_' + str(smother) + '_'
    D1_features = ['race', 'sex', 'maritalstatus']
    Y1_features = ['income']
    X1_features = ['hoursperweek', 'education']

    D2_features = ['age']
    Y2_features = ['income']
    X2_features = ['education', 'occupation', 'hoursperweek']

    D1 = [D1_features, Y1_features, X1_features]
    D2 = [D2_features, Y2_features, X2_features]
    indeps = [D1, D2]


    protected=['sex']
    new_test_rep_str(indeps, features, protected, Y_features, path1,    path2, k=3, classifier=classifier,method='original')
    #train=read_from_csv('/Users/babakmac/Documents/FairDB/data/randomDAG.csv')
    #test=read_from_csv('/Users/babakmac/Documents/FairDB/data/randomDAGtest.csv')
    classification(train, test, ['A', 'B'], ['C'], classifier='log_reg')
