import numpy as np
import pandas as pd
from sklearn import preprocessing

print(__doc__)
from os import chdir
import numpy as np
from Modules.MatrixOprations.contin_table import *
import matplotlib as mpl
import numpy as np
from scipy.stats import norm as mynorm, chi2
mpl.use('TkAgg')
from itertools import cycle
import pandas as pd
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy import interp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import cycle
from os import chdir
#chdir("/Users/babakmac/Documents/fainess/Caputine")
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from itertools import cycle
import pandas as pd
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


print(__doc__)
import sys
from Core.indep_repair import Repair
from Core.Log_Reg_Classifier import *

from Modules.MatrixOprations.contin_table import *
from Modules.InformationTheory.info_theo import *
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#chdir("/Users/babakmac/Documents/HypDB/relational-causal-inference/source/HypDB")
#from core.cov_selection import *
#from core.explanation import *
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC as SKLearn_SVM
from sklearn.tree import DecisionTreeClassifier as SKLearn_DT
from sklearn.naive_bayes import GaussianNB as SKLearn_NB
from sklearn.svm import NuSVC
#chdir("/Users/babakmac/Documents/HypDB/relational-causal-inference/source/HypDB")
#from core.cov_selection import *
#from core.explanation import *


import pprint
from os import chdir
from sklearn.ensemble import RandomForestClassifier
import sys
#sys.path.insert(0, '/Users/babakmac/Documents/relational-causal-inference/source/FairDB')
#from core.cov_selection import *
#from core.explanation import *
#import core.query as sql
#import modules.statistics.cit as ci_test
#from Modules.InformationTheory.info_theo import *
from sklearn.metrics import confusion_matrix
import copy
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
from itertools import combinations
#sys.path.insert(0, '/Users/babakmac/Documents/HypDB/relational-causal-inference/source/HypDB/core')
#from core.cov_selection import *



def list_of_combs(arr):
    """returns a list of all subsets of a list"""

    combs = []
    for i in range(0, len(arr) + 1):
        listing = [list(x) for x in combinations(arr, i)]
        combs.extend(listing)
    return combs

def prec(df,att,Y_features,X_features,inadmisable,ground,ground_val):
    sub_sets = list_of_combs(inadmisable)
    dood = 0
    df=df[df[ground]==ground_val]
    for adm_set in sub_sets:
        grouped=df.groupby(X_features+adm_set)
        total=len(df.index)
        total=0
        bias = 0
        for name, group in grouped:

            mean=group.groupby(att)[Y_features].mean()
            if len(mean.values)==2:
                    val=(mean.values[0]-mean.values[1])[0]
                    if val<0:
                        val=-1*val
                    if val > -1:
                        size = len(group.index)
                        #print(mean,size)
                        total=total+size
                        bias=bias+(val*size)

        if bias>dood:
            dood=float(bias)/total
    return float(dood)

def CP(df,X,x,Y,y):
    i=0
    for item in Y:
        df = df[df[item] == y[i]]
       # print(len(df.index))
        i=i+1
    p_x=len(df.index)
    i = 0
    #print(df[X])
    for item in X:
        df = df[df[item] == x[i]]
        i = i + 1
    p_xy = len(df.index)
    if p_x==0:
        return 0
    return p_xy/p_x




def Hartung(p, L=None, kappa=0.2, alpha=0.10):
    '''
     This function applies the modified inverse normal method for the combination of dependent p-values.
     Arguments:
         p:         vector of p-values.
         lambda:    vector of weights. It must be of the same length of p.
         kappa:     adjustment parameter. It is a positive value (0.2 is the default value),
                    then it is computed as in Hartung, p. 853.
         alpha:     level for the 1-alpha confidence interval for rho (0.10 is the default).

    Returns:
         Value: {"statistic": the Ht test statistic
        	     "parameter": the number of combined tests (p-values)
        	     "pvalue": the combined test p-value
        	     "conf_int": the confidence interval for the estimated correlation
        	     "estimate": the estimated correlation
        	     "null_value": the specified hypothesized value under the null
        	     "alternative": string describing the alternative hypothesis
        	     "method": string indicating the type of combination test (only Z-test is implemented)}
     Reference:
     Hartung, J. (1999): "A note on combining dependent tests of significance",
                         Biometrical Journal, 41(7), 849--855.
    '''

    if L == None:
        L = np.ones(len(p), dtype=float)
    t = mynorm.ppf(p)
    n = float(len(p))
    avt = np.sum(t) / n
    q = np.sum((t - avt) ** 2) / (n - 1)  # Hartung, eqn. (2.2)
    rhohat = 1 - q
    rhostar = max(-1 / (n - 1), rhohat)  # Hartung, p. 851
    if kappa == "formula": kappa = (1 + 1 / (n - 1) - rhostar) / 10  # Hartung, p. 853
    if kappa == "formula2": kappa = (1 + 1 / (n - 1) - rhostar) / 5  # Hartung, p. 853

    # Hartung inverse normal corrected. See eqn. (2.4)

    Ht = np.sum(L * t) / np.sqrt(
        np.sum(L ** 2) + ((np.sum(L)) ** 2 - np.sum(L ** 2)) * (rhostar + kappa * np.sqrt(2 / (n - 1)) * (1 - rhostar)))
    lower = 1 - (n - 1) / chi2.ppf(alpha / 2, (n - 1)) * q
    upper = 1 - (n - 1) / chi2.ppf((1 - alpha / 2), (n - 1)) * q  # Hartung, eqn. (2.3)

    output = dict(statistic=Ht,
                  parameter=n,
                  pvalue=mynorm.cdf(Ht),
                  conf_int=(lower, upper),
                  estimate=rhohat,
                  null_value="Ht=0",
                  alternative="less",
                  method="modified inverse normal combination")

    return output

def adult_intev(df,att,Y_features,X_features,inadmisable,priv,minor,positive):
        total=len(df.index)
        total=0
        bias = 0
        val=0
        n=len(df.index)
        observed=0
        avg=0
        effect=0
        n=0
        overlaped=0
        i=0
        grouped = df.groupby(inadmisable+X_features)
        for name, group in grouped:
            name=list(name)
            count=group.groupby(att)[Y_features[0]].count()
            if len(count.values)>1:
                if count.values[0]>1 and count.values[1]>1:
                    print(name, len(group.index),count.values)
                    if i==0:
                        overlaped=group
                        i=1
                    else:
                        overlaped=pd.concat([group,overlaped])
        df=overlaped

        D=df[X_features].drop_duplicates()
        D=D.values

        G=df[att].drop_duplicates()
        G=G.values

        Q=df[inadmisable].drop_duplicates()
        Q=Q.values
        n=0
        avg_bias=0
        i = 0
        for d in D:
            j = 0
            flag=0
            j = 0
            bias = [0, 0]
            i = 0
            # print(df[X])
            df1=df
            i=0
            for item in X_features:
                df1 = df1[df1[item] == d[i]]
                i = i + 1
            freq_d=len(df1.index)
            bias_minor = 0
            bias_major = 0
            for g in G:
                effect=0
                for q in Q:
                    flag = 0
                    y_dng=CP(df, Y_features, [1],[att]+inadmisable+X_features, [g]+list(q)+list(d))
                    y_mag=CP(df, inadmisable, list(q),[att], [g])
                    if y_dng<0 or y_mag<0:
                        flag=1
                    else:
                        effect=effect+y_dng*y_mag
                if flag!=0:
                    bias[j]=-1
                    print("ERRRRRRORRRRRRR$$$$$$$$$$$$$$$$$$$")
                else:
                    bias[j] = effect
                #if positive ==0:
                #    bias[j]=1-bias[j]
                if g==minor:
                    bias_minor=bias[j]
                if g == priv:
                    bias_major = bias[j]
                j=j+1
                print(g,bias)
            #maxx=max(bias[0],bias[1])
            #minn = min(bias[0], bias[1])
            n=n+1
            bias=0
            minn=min(bias_minor,bias_major)
            maxx = max(bias_minor , bias_major)
            if bias_major!=0:
                bias=(minn/maxx)
                print(bias,'  ',freq_d)
                avg_bias=avg_bias+bias*freq_d
            else:
              avg_bias=avg_bias+1*freq_d
        print(avg_bias/len(df.index))
        return float(avg_bias)/len(df.index)


def ROD(df,att,Y_features,X_features,inadmisable,priv,minor,positive=1):
    sub_sets = list_of_combs(inadmisable)
    dood = 1.1
    for adm_set in sub_sets:
        grouped=df.groupby(X_features+adm_set)
        total=len(df.index)
        total=0
        bias = 0
        val=0
        observed=0
        for name, group in grouped:
            mean=group.groupby(att)[Y_features].mean()
            count = group.groupby(att)[Y_features].count()
            size = len(group.index)
            #print(mean)
            if size>0:
                if len(mean.values)==2:
                            min_id=0
                            priv_eff=0
                            if mean.values[0]==mean.values[1]:
                                val=1
                            else:
                                if mean.index[0]==priv:
                                    priv_eff=mean.values[0]
                                    min_id=1

                                elif mean.index[1]==priv:
                                    priv_eff=mean.values[1]
                                    min_id=0
                                min_eff=mean.values[min_id]
                                max_ratio=max(mean.values[0],mean.values[1])
                                min_ratio=min(mean.values[0],mean.values[1])
                                if positive==0:
                                    min_eff=1-min_eff
                                    priv_eff=1-priv_eff

                                val = (min_ratio) / (max_ratio)
                            #print(val)
                            size = len(group.index)
                            #print(val,size)
                            total=total+size
                            bias=bias+(val*size)
                            observed=1

        #print('##########')
        #print(bias )
        if observed:
            if total!=0:
               #print(bias/total)
               bias=float(bias)/total
            if bias<dood:
                    dood=float(bias)
    print(dood)
    return float(dood)

def weighted_median(values, weights):
    ''' compute the weighted median of values list. The
weighted median is computed as follows:
    1- sort both lists (values and weights) based on values.
    2- select the 0.5 point from the weights and return the corresponding values as results
    e.g. values = [1, 3, 0] and weights=[0.1, 0.3, 0.6] assuming weights are probabilities.
    sorted values = [0, 1, 3] and corresponding sorted weights = [0.6,     0.1, 0.3] the 0.5 point on
    weight corresponds to the first item which is 0. so the weighted     median is 0.'''

    #convert the weights into probabilities
    sum_weights = sum(weights)
    weights = np.array([(w*1.0)/sum_weights for w in weights])
    #sort values and weights based on values
    values = np.array(values)
    sorted_indices = np.argsort(values)
    values_sorted  = values[sorted_indices]
    weights_sorted = weights[sorted_indices]
    #select the median point
    it = np.nditer(weights_sorted, flags=['f_index'])
    accumulative_probability = 0
    median_index = -1
    while not it.finished:
        accumulative_probability += it[0]
        if accumulative_probability > 0.5:
            median_index = it.index
            return values_sorted[median_index]
        elif accumulative_probability == 0.5:
            median_index = it.index
            it.iternext()
            next_median_index = it.index
            return np.mean(values_sorted[[median_index, next_median_index]])
        it.iternext()

    return values_sorted[median_index]


def OR(df,att,Y_features,X_features,inadmisable,priv,minor,positive=1):
    sub_sets = list_of_combs(inadmisable)
    dood = 1.1
    for adm_set in sub_sets:
        grouped=df.groupby(X_features+adm_set)
        total=len(df.index)
        total=0
        bias = []
        val=0
        observed=0
        pvaluea=0
        pvalue=0
        i=0
        prob=[]
        tables=[]
        for name, group in grouped:
            count=group.groupby(att).count()
            count=count.values
            tbl = ContinTable()
            tbl.data_to_cnt(group, [att], [Y_features])
            table=tbl.matrix
            size=len(group.index)
            if len(count)>=2:
                if (count[0])[0]>5 and (count[1])[0]>5:
                    table2 = sm.stats.Table.from_data(group[[att,Y_features]])
                    #print(tbl.table)
                    if table.shape==(2,2):
                            tables.insert(0,table2.table+1)
                            #print(table2.table)

                    else:
                        tables.insert(0,np.asarray([[float(size/4), float(size/4)], [float(size/4), float(size/4)]]))


        orr=1
        pval=1
        print('##########')
        if len(tables)>1:
            st = sm.stats.StratifiedTable(tables)
            print(st.summary())
            orr=st.oddsratio_pooled
            pval=st.test_null_odds()
        '''
        bias=   [float(x) for x in bias]
        prob = [float(x) / sum(prob) for x in prob]
        print(sum(prob) )
        i=0
        orr=0
        for item in bias:
            orr=orr+item*prob[i]
            i=i+1
        #bias=weighted_median(bias,prob)
        print(orr)
        '''
        return orr,pval








def CDI(df,att,Y_features,X_features,inadmisable,priv,minor,positive=1):
    sub_sets = list_of_combs(inadmisable)
    dood = 1.1
    for adm_set in sub_sets:
        grouped=df.groupby(X_features+adm_set)
        total=len(df.index)
        total=0
        bias = 0
        val=0
        observed=0
        for name, group in grouped:
            mean=group.groupby(att)[Y_features].mean()
            count = group.groupby(att)[Y_features].count()
            size = len(group.index)
            print(mean)
            min_id = 0
            priv_eff = 0
            if size>0:
                if len(mean.values)==2:
                            min_id=0
                            priv_eff=0
                            if mean.values[0]==mean.values[1]:
                                val=1
                            else:
                                if mean.index[0]==priv:
                                    priv_eff=mean.values[0]
                                    min_id=1

                                elif mean.index[1]==priv:
                                    priv_eff=mean.values[1]
                                    min_id=0
                                min_eff=mean.values[min_id]
                            if priv_eff>0 and  positive==1:
                                val = (min_eff) / (priv_eff)
                            else:
                                val = (priv_eff) / (min_eff)
                            print(val)
                            #if val>1:
                            #    val=1
                            size = len(group.index)
                            print(val,size)
                            total=total+size
                            bias=bias+(val*size)
                            observed=1

        print('##########')
        print(bias )
        if total!=0:
               print(bias/total)
               bias=float(bias)/total
    print(bias)
    return float(bias)

def data_split(data,outcome,path,k=5,test_size=0.3):


    rs = StratifiedShuffleSplit(n_splits=k, test_size=test_size, random_state=2)
    data_y = pd.DataFrame(data[outcome])
    data_X = data.drop([outcome], axis=1)
    rs.get_n_splits(data_X, data_y)
    j = 0
    for test, train in rs.split(data_X,data_y):
        cur_test  = data.iloc[train]
        cur_train = data.iloc[test]
        cur_train.to_csv(path + 'train_' + str(j) + '.csv', encoding='utf-8', index=False)
        print(len(cur_train.index))
        print(path + 'train_' + str(j) + '.csv')
        cur_test.to_csv(path + 'test_' + str(j) + '.csv', encoding='utf-8', index=False)
        print(len(cur_test.index))
        print(path + 'test_' + str(j) + '.csv')
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


def classification_rep_str(D_features, Y_features, X_features, path1, path2, k=5, droped=False, classifier='log_reg'):
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



def old_classification_rep_str(indeps, features, protecteds, Y_features, path1, path2, k=5, droped=False, classifier='log_reg', method='original'):
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


def classification_from_files(features, Y_features, path1, path2, classifier,k=5, method='original',iteration=100,plot=False):
    print('Traning on :', features)
    clf=classifier
    if Y_features[0] in features:
        features.remove(Y_features[0])

    D_features=[]
    X_features=features
    tprs = []
    aucs = []
    accur = []
    mean_fpr = np.linspace(0, 1, 100)
    for j in range(0, k):
        print(path1+str(j)+'.csv')
        cur_train=read_from_csv(path1+str(j)+'.csv')
        print(path2+str(j)+'.csv')
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

        #sc = StandardScaler()
        #train_X = sc.fit_transform(train_X)
        #train_y = sc.transform(train_y)

        #if droped:
        #    dft4 = pd.get_dummies(cur_test[X_features])
        #else:
        #    dft4 = pd.get_dummies(cur_test[ D_features+X_features])
        #print(cur_test[D_features+X_features])






        dft5 = pd.get_dummies(cur_test[Y_features])
        y_pred=[]
        # logit = sm.Logit(train_df['bscore'], train_df['juv_misd_count'])
        X = dft4.values
        y = dft5.values
        y = y.flatten()

        #print("#####################",len(train_X),len(train_y),type(train_X),type(train_y),train_X,train_y,X.shape)
        print(X.shape,train_X.shape)
        #logit = sm.Logit(train_X,cur_train[Y_features[0]])

        if classifier == 'NN':
            classifier = Sequential()
            # Adding the input layer and the first hidden layer
            classifier.add(Dense(output_dim=train_X.shape[1], init='uniform', activation='relu', input_dim=train_X.shape[1]))
            # Adding the second hidden layer
            classifier.add(Dense(output_dim=train_X.shape[1], init='uniform', activation='relu'))
            # Adding the output layer
            classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            classifier.fit(train_X, train_y, batch_size=10, nb_epoch=iteration)
            probas_ = classifier.predict(X)
            y_pred = (probas_ > 0.5)
            classifier ='NN'
            clas_name='NN'

        else:
            try:
                probas_ = classifier.fit(train_X, train_y).predict_proba(X)
                probas_=probas_[:, 1]
                y_pred = classifier.predict(X)
                clas_name=clf.__class__.__name__
            except:
                return

        scoring = 'accuracy'
        #result = logit.fit()


        #print(probas_)
        #predicted = cross_validation.cross_val_predict(logreg, X, y, cv=10)

        cur_test.insert(0,'y',y_pred)    # insert the outcome into the test dataset
        #cur_test['FP']=cur_test.loc[(cur_test[Y_features] ==1) & (cur_test.y == 1)]
        cur_test['FP'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 0 and x['y'] == 1 else 0, axis=1)
        cur_test['FN'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 1 and x['y'] == 0 else 0, axis=1)
        cur_test['TP'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 1 and x['y'] == 1 else 0, axis=1)
        cur_test['TN'] = cur_test.apply(lambda x: 1 if x[Y_features[0]] == 0 and x['y'] == 0 else 0, axis=1)
        score_dt = accuracy_score(cur_test[Y_features[0]], y_pred)
        aucssss=roc_auc_score(cur_test[Y_features[0]], probas_)

        i=0
        Z=[]
        fpr, tpr, thresholds = roc_curve(cur_test[Y_features[0]], probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(aucssss)
        accur.append(score_dt)
        print('###################')
        print(clas_name, score_dt,aucssss)
        print('###################')

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
        if method=='dropped':
            file=path1+ '_dropped_'+clas_name+'_'+str(j)+'.csv'
        else:
            file=path1+ '_'+clas_name+'_'+str(j)+'.csv'
        df.to_csv(file)
        print(file)
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    if plot:
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
    return aucs,accur

if __name__ == "__main__":
    #df0 = read_from_csv("/Users/babakmac/Documents/fainess/Caputine/Data/COMPAS/train__LogisticRegression_1.csv")
    df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train__MLPClassifier_1.csv")
    df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train__repsat_300__MLPClassifier_1.csv")
    #df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train__repsat_300__LogisticRegression_4.csv")
    #df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train__LogisticRegression_4.csv")
    #df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train__repsat_300_0.csv")
    #df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train__repIC_0.csv")
    #df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train_0.csv")
    df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train_0.csv")
    #df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train__repIC_0.csv")
    df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train__repsat_300_0.csv")
    #df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train__repsat_300__LogisticRegression_4.csv")
    #df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train__repMF_0.csv")
    #df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train_0.csv")
    #df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train_0.csv")
    #df0 = read_from_csv("/Users/babakmac/Documents/fainess/Backup/fainess/Caputine/Data/Adult_test/train_0.csv")
    #df0 = df0.sample(n=3000)
    D_features = ['sex', 'maritalstatus','age']
    Y_features = ['income']
    X_features = ['hoursperweek','occupation','capitalgain','workclass','education']
    inadmissable=['maritalstatus','age']
    #inadmissable=[]
    D = [D_features, Y_features, X_features]
    indeps = [D]


    #0.3615240296561736
    df0 = read_from_csv("/Users/babakmac/Documents/Data/Adult/train__repsat_300_0.csv")
    D_features = ['race']
    Y_features = ['is_recid']
    X_features = ['priors_count','age_cat']
    inadmissable = [ 'age_cat']
    #df = read_from_csv("/Users/babakmac/Documents/fainess/Capuchin/Experiments/Nips/compas/experiment_data1/train__repsat_10__RandomForestClassifier_4.csv")
    #adult_intev(df0, 'race', ['y'], X_features, inadmissable)
    df = read_from_csv("/Users/babakmac/Documents/fainess/Capuchin/Experiments/Nips/compas/experiment_data1/train__repsat_20__RandomForestClassifier_1.csv")
    #df = read_from_csv("/Users/babakmac/Documents/fainess/Capuchin/Experiments/Nips/compas/experiment_data1/train_0.csv")
    #df = read_from_csv("/Users/babakmac/Documents/fainess/Capuchin/Experiments/Nips/compas/experiment_data1/train__RandomForestClassifier_2.csv")
    #df = read_from_csv("/Users/babakmac/Documents/fainess/Capuchin/Experiments/Nips/compas/experiment_data1/train__repsat_8_0.csv")
    #x=adult_intev(df0, 'race', ['y'], X_features, inadmissable)
    #print('Me',x)
    mynorm.ppf([0.001, 0.5, 0.999])
    att = 'age_cat'
    priv = '25 to 45'
    minor  = 'African-American'
    priv = 'Caucasian'
    #df = df.replace(priv, 1)
    #df = df.replace(minor, 1)
    #df = df.replace(other, 0)
    #dood=Dood(df, 'race', ['y'], X_features, [],priv,minor,positive=0)
    #print('here',dood)
    #x=adult_intev(df, 'race', ['y'], ['priors_count','c_charge_degree','age_cat'],['age_cat'],priv,minor,positive=0)
    #print('here', x)
    data1 = pd.read_csv("/Users/babakmac/Documents/XDBData/compasFN.csv")
    data1 = data1[data1['score_text'].isin(['Low', 'High'])]
    data1 = data1[data1['race'].isin(['African-American', 'Caucasian'])]
    data1 = data1.replace('Low', 0)
    data1 = data1.replace('High', 1)
    #dood=Dood2(data1, 'race', 'score_text', ['age_cat', 'priors_count'], [], 'Caucasian',
    #     'African-American', positive=0)
    #print(dood)
    #data1['score_text']=1-data1['score_text']
    #data1['is_recid'] = 1 - data1['is_recid']
    Hartung([0.1,0.5])
    D_features = ['Gender']
    Y_features = ['Income Binary']
    X_features = ['Education Years', 'Age (decade)']
    data1=read_from_csv('/Users/babakmac/Documents/fainess/Capuchin/Experiments/Nips/compas/experiment_data1/train__repsat_30__LogisticRegression_0.csv')
    D_features = ['race']
    Y_features = ['is_recid']
    X_features = ['priors_count', 'c_charge_degree', 'age_cat']
    #D_features = ['Gender']
    #Y_features = ['Income Binary']
    #X_features = ['Education Years', 'Age (decade)']
    #rod=OR(data1, 'race', 'y', X_features, [], 'Caucasian', 'African-American',positive=1)
    #print(rod)
    data1 = pd.read_csv("/Users/babakmac/Documents/XDBData/compasFN.csv")
    data1 = data1[data1['score_text'].isin(['Low', 'High'])]
    data1 = data1[data1['race'].isin(['African-American', 'Caucasian'])]
    data1 = data1.replace('Low', 0)
    data1 = data1.replace('High', 1)
    D_features = ['race', 'sex']
    Y_features = ['is_recid']
    X_features = ['priors_count', 'c_charge_degree', 'age_cat']
    #rod,pv=OR(data1, 'race', 'score_text', X_features, [], 'Caucasian', 'African-American',positive=1)
    ##print(rod)
    #rod,pv=OR(data1, 'race', 'is_recid', X_features, [], 'Caucasian', 'African-American',positive=1)
    #print(rod)


    data1 = pd.read_csv("/Users/babakmac/Documents/fainess/Backup/fainess/Caputine/Data/COMPAS_test_option/train__repnaive_0.csv")
    D_features = ['race', 'sex']
    Y_features = ['is_recid']
    X_features = ['priors_count', 'c_charge_degree', 'age_cat']
    rod,pv=OR(data1, 'race', 'is_recid', X_features, [], 'Caucasian', 'African-American',positive=1)
    print(rod,pv)





    #classification_from_files(indeps, features, protected, Y_features, path1,    path2, k=3, classifier=classifier,method='original')
