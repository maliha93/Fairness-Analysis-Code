
print(__doc__)
from os import chdir
chdir("/Users/babakmac/Documents/fainess/FairDB")
#import numresultpy as np
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
from cvxpy import *
print(__doc__)
from Core.indep_repair import Repair
from Core.Log_Reg_Classifier import *
from Modules.InformationTheory.info_theo import *
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from  Core.grounder  import advanced_grounder,data_from_sat,grounder4
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
from Modules.MatrixOprations.contin_table import *
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











if __name__ == '__main__':
  print(os.getcwd())

