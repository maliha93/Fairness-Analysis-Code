import numpy as np
from collections import namedtuple
import pandas as pd
import sys
from metric import metric, cd
from helper import make_dataset

class Model(namedtuple('Model', 'pred label')):
    def logits(self):
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        return len(self.pred)

    def base_rate(self):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(self.label)

    def accuracy(self):
        return self.accuracies().mean()

    def precision(self):
        return (self.label[self.pred.round() == 1]).mean()

    def recall(self):
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        True positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))

    def fpr(self):
        """
        False positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))

    def tnr(self):
        """
        True negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))

    def fnr(self):
        """
        False negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))

    def fn_cost(self):
        """
        Generalized false negative cost
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Generalized false positive cost
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        return self.pred.round() == self.label

    def calib_eq_odds(self, other, fp_rate, fn_rate, mix_rates=None):
        if fn_rate == 0:
            self_cost = self.fp_cost()
            other_cost = other.fp_cost()
            #print(self_cost, other_cost)
            self_trivial_cost = self.trivial().fp_cost()
            other_trivial_cost = other.trivial().fp_cost()
        elif fp_rate == 0:
            self_cost = self.fn_cost()
            other_cost = other.fn_cost()
            self_trivial_cost = self.trivial().fn_cost()
            other_trivial_cost = other.trivial().fn_cost()
        else:
            self_cost = self.weighted_cost(fp_rate, fn_rate)
            other_cost = other.weighted_cost(fp_rate, fn_rate)
            self_trivial_cost = self.trivial().weighted_cost(fp_rate, fn_rate)
            other_trivial_cost = other.trivial().weighted_cost(fp_rate, fn_rate)

        other_costs_more = other_cost > self_cost
        self_mix_rate = (other_cost - self_cost) / (self_trivial_cost - self_cost) if other_costs_more else 0
        other_mix_rate = 0 if other_costs_more else (self_cost - other_cost) / (other_trivial_cost - other_cost)

        # New classifiers
        self_indices = np.random.permutation(len(self.pred))[:int(self_mix_rate * len(self.pred))]
        self_new_pred = self.pred.copy()
        self_new_pred[self_indices] = self.base_rate()
        calib_eq_odds_self = Model(self_new_pred, self.label)

        other_indices = np.random.permutation(len(other.pred))[:int(other_mix_rate * len(other.pred))]
        other_new_pred = other.pred.copy()
        other_new_pred[other_indices] = other.base_rate()
        calib_eq_odds_other = Model(other_new_pred, other.label)

        if mix_rates is None:
            return calib_eq_odds_self, calib_eq_odds_other, (self_mix_rate, other_mix_rate)
        else:
            return calib_eq_odds_self, calib_eq_odds_other

    def trivial(self):
        """
        Given a classifier, produces the trivial classifier
        (i.e. a model that just returns the base rate for every prediction)
        """
        base_rate = self.base_rate()
        pred = np.ones(len(self.pred)) * base_rate
        return Model(pred, self.label)

    def weighted_cost(self, fp_rate, fn_rate):
        """
        Returns the weighted cost
        If fp_rate = 1 and fn_rate = 0, returns self.fp_cost
        If fp_rate = 0 and fn_rate = 1, returns self.fn_cost
        If fp_rate and fn_rate are nonzero, returns fp_rate * self.fp_cost * (1 - self.base_rate) +
            fn_rate * self.fn_cost * self.base_rate
        """
        norm_const = float(fp_rate + fn_rate) if (fp_rate != 0 and fn_rate != 0) else 1
        res = fp_rate / norm_const * self.fp_cost() * (1 - self.base_rate()) + \
            fn_rate / norm_const * self.fn_cost() * self.base_rate()
        return res

    def __repr__(self):
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'T.P. rate:\t%.3f' % self.tpr(),
            'T.N. rate:\t%.3f' % self.tnr(),
            'Precision:\t%.3f' % self.precision(),
            'Recall:\t\t%.3f'    % self.recall(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])



def Adult(f="data/adult_post.csv"):
    
    data_filename = f
    test_and_val_data = pd.read_csv(data_filename)

    order = np.arange(len(test_and_val_data))
    leng = order.shape[0]
    val_indices = order[0:int(leng*0.7)]
    test_indices = order[int(leng*0.7):]
    val_data = test_and_val_data.iloc[val_indices]
    test_data = test_and_val_data.iloc[test_indices]

    # Create model objects - one for each group, validation and test
    group_0_val_data = val_data[val_data['group'] == 0]
    group_1_val_data = val_data[val_data['group'] == 1]
    group_0_test_data = test_data[test_data['group'] == 0]
    group_1_test_data = test_data[test_data['group'] == 1]

    group_0_val_model = Model(group_0_val_data['prediction'].to_numpy(), group_0_val_data['label'].to_numpy())
    group_1_val_model = Model(group_1_val_data['prediction'].to_numpy(), group_1_val_data['label'].to_numpy())
    group_0_test_model = Model(group_0_test_data['prediction'].to_numpy(), group_0_test_data['label'].to_numpy())
    group_1_test_model = Model(group_1_test_data['prediction'].to_numpy(), group_1_test_data['label'].to_numpy())

    # Find mixing rates for equalized odds models
    _, _, mix_rates = Model.calib_eq_odds(group_0_val_model, group_1_val_model, 0, 1)

    # Apply the mixing rates to the test models
    calib_eq_odds_group_0_test_model, calib_eq_odds_group_1_test_model = Model.calib_eq_odds(group_0_test_model,
                                                                                             group_1_test_model,
                                                                                             0, 1,
                                                                                             mix_rates)
    cd_calib_eq_odds_group_0_test_model, cd_calib_eq_odds_group_1_test_model = Model.calib_eq_odds(group_1_test_model,
                                                                                             group_0_test_model,
                                                                                             0, 1,
                                                                                             mix_rates)
    
    metric(calib_eq_odds_group_0_test_model.label, calib_eq_odds_group_0_test_model.pred,
           calib_eq_odds_group_1_test_model.label, calib_eq_odds_group_1_test_model.pred)
    y_cd = cd(calib_eq_odds_group_0_test_model.pred.round(), calib_eq_odds_group_1_test_model.pred.round(),\
       cd_calib_eq_odds_group_0_test_model.pred.round(),cd_calib_eq_odds_group_1_test_model.pred.round())
    
    group_0_test_data['pred'] = calib_eq_odds_group_0_test_model.pred.round()
    group_1_test_data['pred'] = calib_eq_odds_group_1_test_model.pred.round()
    df = group_0_test_data.append(group_1_test_data)
    df = df.drop(['group', 'prediction', 'label'], axis=1).sample(frac=1)
    df.to_csv("results_Pleiss/adult_test_repaired.csv", index=False) 
    np.savetxt("results_Pleiss/adult_test_repaired_cd.csv", y_cd, delimiter=",")
           
def Compas(f="data/compas_post.csv", f1='', f2=''):

    data_filename = f
    test_and_val_data = pd.read_csv(data_filename)

    order = np.arange(len(test_and_val_data))
    leng = order.shape[0]
    val_indices = order[0:int(leng*0.7)]
    test_indices = order[int(leng*0.7):]
    val_data = test_and_val_data.iloc[val_indices]
    test_data = test_and_val_data.iloc[test_indices]

    # Create model objects - one for each group, validation and test
    group_0_val_data = val_data[val_data['group'] == 0]
    group_1_val_data = val_data[val_data['group'] == 1]
    group_0_test_data = test_data[test_data['group'] == 0]
    group_1_test_data = test_data[test_data['group'] == 1]

    group_0_val_model = Model(group_0_val_data['prediction'].to_numpy(), group_0_val_data['label'].to_numpy())
    group_1_val_model = Model(group_1_val_data['prediction'].to_numpy(), group_1_val_data['label'].to_numpy())
    group_0_test_model = Model(group_0_test_data['prediction'].to_numpy(), group_0_test_data['label'].to_numpy())
    group_1_test_model = Model(group_1_test_data['prediction'].to_numpy(), group_1_test_data['label'].to_numpy())

    # Find mixing rates for equalized odds models
    _, _, mix_rates = Model.calib_eq_odds(group_0_val_model, group_1_val_model, 1, 1)

    # Apply the mixing rates to the test models
    calib_eq_odds_group_0_test_model, calib_eq_odds_group_1_test_model = Model.calib_eq_odds(group_0_test_model,
                                                                                             group_1_test_model,
                                                                                             1, 1,
                                                                                             mix_rates)
    
    cd_calib_eq_odds_group_0_test_model, cd_calib_eq_odds_group_1_test_model = Model.calib_eq_odds(group_1_test_model,
                                                                                             group_0_test_model,
                                                                                             1, 1,
                                                                                             mix_rates)
    
    metric(calib_eq_odds_group_0_test_model.label, calib_eq_odds_group_0_test_model.pred,
           calib_eq_odds_group_1_test_model.label, calib_eq_odds_group_1_test_model.pred)
    y_cd = cd(calib_eq_odds_group_0_test_model.pred.round(), calib_eq_odds_group_1_test_model.pred.round(),\
       cd_calib_eq_odds_group_0_test_model.pred.round(),cd_calib_eq_odds_group_1_test_model.pred.round())
    
    group_0_test_data['pred'] = calib_eq_odds_group_0_test_model.pred.round()
    group_1_test_data['pred'] = calib_eq_odds_group_1_test_model.pred.round()
    df = group_0_test_data.append(group_1_test_data)
    df = df.drop(['group', 'prediction', 'label'], axis=1).sample(frac=1)
    df.to_csv(f1+"results_Pleiss/compas_test_repaired"+f2+".csv", index=False)
    np.savetxt(f1+"results_Pleiss/compas_test_repaired"+f2+"_cd.csv", y_cd, delimiter=",")       
           
def German(f="data/german_post.csv"):

    data_filename = f
    test_and_val_data = pd.read_csv(data_filename)

    order = np.arange(len(test_and_val_data))
    leng = order.shape[0]
    val_indices = order[0:int(leng*0.7)]
    test_indices = order[int(leng*0.7):]
    val_data = test_and_val_data.iloc[val_indices]
    test_data = test_and_val_data.iloc[test_indices]

    # Create model objects - one for each group, validation and test
    group_0_val_data = val_data[val_data['group'] == 0]
    group_1_val_data = val_data[val_data['group'] == 1]
    group_0_test_data = test_data[test_data['group'] == 0]
    group_1_test_data = test_data[test_data['group'] == 1]

    group_0_val_model = Model(group_0_val_data['prediction'].to_numpy(), group_0_val_data['label'].to_numpy())
    group_1_val_model = Model(group_1_val_data['prediction'].to_numpy(), group_1_val_data['label'].to_numpy())
    group_0_test_model = Model(group_0_test_data['prediction'].to_numpy(), group_0_test_data['label'].to_numpy())
    group_1_test_model = Model(group_1_test_data['prediction'].to_numpy(), group_1_test_data['label'].to_numpy())

    # Find mixing rates for equalized odds models
    _, _, mix_rates = Model.calib_eq_odds(group_0_val_model, group_1_val_model, 1, 1)

    # Apply the mixing rates to the test models
    calib_eq_odds_group_0_test_model, calib_eq_odds_group_1_test_model = Model.calib_eq_odds(group_0_test_model,
                                                                                             group_1_test_model,
                                                                                             1, 0,
                                                                                             mix_rates)

    cd_calib_eq_odds_group_0_test_model, cd_calib_eq_odds_group_1_test_model = Model.calib_eq_odds(group_1_test_model,
                                                                                             group_0_test_model,
                                                                                             1, 0,
                                                                                             mix_rates)
    
    metric(calib_eq_odds_group_0_test_model.label, calib_eq_odds_group_0_test_model.pred,
           calib_eq_odds_group_1_test_model.label, calib_eq_odds_group_1_test_model.pred)
    y_cd = cd(calib_eq_odds_group_0_test_model.pred.round(), calib_eq_odds_group_1_test_model.pred.round(),\
       cd_calib_eq_odds_group_0_test_model.pred.round(),cd_calib_eq_odds_group_1_test_model.pred.round())
    
    group_0_test_data['pred'] = calib_eq_odds_group_0_test_model.pred.round()
    group_1_test_data['pred'] = calib_eq_odds_group_1_test_model.pred.round()
    df = group_0_test_data.append(group_1_test_data)
    df = df.drop(['group', 'prediction', 'label'], axis=1).sample(frac=1)
    df.to_csv("results_Pleiss/german_test_repaired.csv", index=False)       
    np.savetxt("results_Pleiss/german_test_repaired_cd.csv", y_cd, delimiter=",")
           
def Pleiss(dataset):
    
    make_dataset(dataset)
    if dataset == 'adult':
        Adult()
    elif dataset == 'compas':
        Compas()
    elif dataset == 'german':
        German()
    elif dataset == 'credit':
        Credit()





