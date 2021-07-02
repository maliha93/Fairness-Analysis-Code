import cvxpy as cvx
import numpy as np
from collections import namedtuple
from metric import metric, cd
import pandas as pd
import sys
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

    def eq_odds(self, othr, mix_rates=None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(othr)
        sp2p, sn2p, op2p, on2p = tuple(mix_rates)

        self_fair_pred = self.pred.copy()
        self_pp_indices, = np.nonzero(self.pred.round())
        self_pn_indices, = np.nonzero(1 - self.pred.round())
        np.random.shuffle(self_pp_indices)
        np.random.shuffle(self_pn_indices)

        n2p_indices = self_pn_indices[:int(len(self_pn_indices) * sn2p)]
        self_fair_pred[n2p_indices] = 1 - self_fair_pred[n2p_indices]
        p2n_indices = self_pp_indices[:int(len(self_pp_indices) * (1 - sp2p))]
        self_fair_pred[p2n_indices] = 1 - self_fair_pred[p2n_indices]

        othr_fair_pred = othr.pred.copy()
        othr_pp_indices, = np.nonzero(othr.pred.round())
        othr_pn_indices, = np.nonzero(1 - othr.pred.round())
        np.random.shuffle(othr_pp_indices)
        np.random.shuffle(othr_pn_indices)

        n2p_indices = othr_pn_indices[:int(len(othr_pn_indices) * on2p)]
        othr_fair_pred[n2p_indices] = 1 - othr_fair_pred[n2p_indices]
        p2n_indices = othr_pp_indices[:int(len(othr_pp_indices) * (1 - op2p))]
        othr_fair_pred[p2n_indices] = 1 - othr_fair_pred[p2n_indices]

        fair_self = Model(self_fair_pred, self.label)
        fair_othr = Model(othr_fair_pred, othr.label)

        if not has_mix_rates:
            return fair_self, fair_othr, mix_rates
        else:
            return fair_self, fair_othr

    def eq_odds_optimal_mix_rates(self, othr):
        sbr = float(self.base_rate())
        obr = float(othr.base_rate())

        sp2p = cvx.Variable(1)
        sp2n = cvx.Variable(1)
        sn2p = cvx.Variable(1)
        sn2n = cvx.Variable(1)

        op2p = cvx.Variable(1)
        op2n = cvx.Variable(1)
        on2p = cvx.Variable(1)
        on2n = cvx.Variable(1)

        sfpr = self.fpr() * sp2p + self.tnr() * sn2p
        sfnr = self.fnr() * sn2n + self.tpr() * sp2n
        ofpr = othr.fpr() * op2p + othr.tnr() * on2p
        ofnr = othr.fnr() * on2n + othr.tpr() * op2n
        error = sfpr + sfnr + ofpr + ofnr

        sflip = 1 - self.pred
        sconst = self.pred
        oflip = 1 - othr.pred
        oconst = othr.pred

        sm_tn = np.logical_and(self.pred.round() == 0, self.label == 0)
        sm_fn = np.logical_and(self.pred.round() == 0, self.label == 1)
        sm_tp = np.logical_and(self.pred.round() == 1, self.label == 1)
        sm_fp = np.logical_and(self.pred.round() == 1, self.label == 0)

        om_tn = np.logical_and(othr.pred.round() == 0, othr.label == 0)
        om_fn = np.logical_and(othr.pred.round() == 0, othr.label == 1)
        om_tp = np.logical_and(othr.pred.round() == 1, othr.label == 1)
        om_fp = np.logical_and(othr.pred.round() == 1, othr.label == 0)

        spn_given_p = (sn2p * (sflip * sm_fn).mean() + sn2n * (sconst * sm_fn).mean()) / sbr + \
                      (sp2p * (sconst * sm_tp).mean() + sp2n * (sflip * sm_tp).mean()) / sbr

        spp_given_n = (sp2n * (sflip * sm_fp).mean() + sp2p * (sconst * sm_fp).mean()) / (1 - sbr) + \
                      (sn2p * (sflip * sm_tn).mean() + sn2n * (sconst * sm_tn).mean()) / (1 - sbr)

        opn_given_p = (on2p * (oflip * om_fn).mean() + on2n * (oconst * om_fn).mean()) / obr + \
                      (op2p * (oconst * om_tp).mean() + op2n * (oflip * om_tp).mean()) / obr

        opp_given_n = (op2n * (oflip * om_fp).mean() + op2p * (oconst * om_fp).mean()) / (1 - obr) + \
                      (on2p * (oflip * om_tn).mean() + on2n * (oconst * om_tn).mean()) / (1 - obr)

        constraints = [
            sp2p == 1 - sp2n,
            sn2p == 1 - sn2n,
            op2p == 1 - op2n,
            on2p == 1 - on2n,
            sp2p <= 1,
            sp2p >= 0,
            sn2p <= 1,
            sn2p >= 0,
            op2p <= 1,
            op2p >= 0,
            on2p <= 1,
            on2p >= 0,
            spp_given_n == opp_given_n,
            spn_given_p == opn_given_p,
        ]

        prob = cvx.Problem(cvx.Minimize(error), constraints)
        prob.solve()

        res = np.array([sp2p.value, sn2p.value, op2p.value, on2p.value])
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
    _, _, mix_rates = Model.eq_odds(group_0_val_model, group_1_val_model)

    # Apply the mixing rates to the test models
    eq_odds_group_0_test_model, eq_odds_group_1_test_model = Model.eq_odds(group_0_test_model,
                                                                           group_1_test_model,
                                                                           mix_rates)
    
    cd_eq_odds_group_0_test_model, cd_eq_odds_group_1_test_model = Model.eq_odds(group_1_test_model,
                                                                           group_0_test_model,
                                                                           mix_rates)
    metric(eq_odds_group_0_test_model.label, eq_odds_group_0_test_model.pred,
           eq_odds_group_1_test_model.label, eq_odds_group_1_test_model.pred)
    y_cd = cd(eq_odds_group_0_test_model.pred.round(), eq_odds_group_1_test_model.pred.round(),\
       cd_eq_odds_group_0_test_model.pred.round(),cd_eq_odds_group_1_test_model.pred.round())
       
    group_0_test_data['pred'] = eq_odds_group_0_test_model.pred.round()
    group_1_test_data['pred'] = eq_odds_group_1_test_model.pred.round()
    df = group_0_test_data.append(group_1_test_data)
    df = df.drop(['group', 'prediction', 'label'], axis=1).sample(frac=1)
    df.to_csv("results_Hardt/adult_test_repaired.csv", index=False) 
    np.savetxt("results_Hardt/adult_test_repaired_cd.csv", y_cd, delimiter=",")
    
       
def Compas(f="data/compas_post.csv", f1='', f2=''):

    data_filename = f
    test_and_val_data = pd.read_csv(data_filename)

    # Randomly split the data into two sets - one for computing the fairness constants
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
    _, _, mix_rates = Model.eq_odds(group_0_val_model, group_1_val_model)

    # Apply the mixing rates to the test models
    eq_odds_group_0_test_model, eq_odds_group_1_test_model = Model.eq_odds(group_0_test_model,
                                                                           group_1_test_model,
                                                                           mix_rates)
    cd_eq_odds_group_0_test_model, cd_eq_odds_group_1_test_model = Model.eq_odds(group_1_test_model,
                                                                           group_0_test_model,
                                                                           mix_rates)
    metric(eq_odds_group_0_test_model.label, eq_odds_group_0_test_model.pred,
           eq_odds_group_1_test_model.label, eq_odds_group_1_test_model.pred)
    y_cd = cd(eq_odds_group_0_test_model.pred.round(), eq_odds_group_1_test_model.pred.round(),\
       cd_eq_odds_group_0_test_model.pred.round(),cd_eq_odds_group_1_test_model.pred.round())
    
    group_0_test_data['pred'] = eq_odds_group_0_test_model.pred.round()
    group_1_test_data['pred'] = eq_odds_group_1_test_model.pred.round()
    df = group_0_test_data.append(group_1_test_data)
    df = df.drop(['group', 'prediction', 'label'], axis=1).sample(frac=1)
    df.to_csv(f1+"results_Hardt/compas_test_repaired"+f2+".csv", index=False)
    np.savetxt(f1+"results_Hardt/compas_test_repaired"+f2+"_cd.csv", y_cd, delimiter=",")
       
           
def German(f="data/german_post.csv"):

    data_filename = f
    test_and_val_data = pd.read_csv(data_filename)

    # Randomly split the data into two sets - one for computing the fairness constants
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
    _, _, mix_rates = Model.eq_odds(group_0_val_model, group_1_val_model)

    # Apply the mixing rates to the test models
    eq_odds_group_0_test_model, eq_odds_group_1_test_model = Model.eq_odds(group_0_test_model,
                                                                           group_1_test_model,
                                                                           mix_rates)

    cd_eq_odds_group_0_test_model, cd_eq_odds_group_1_test_model = Model.eq_odds(group_1_test_model,
                                                                           group_0_test_model,
                                                                           mix_rates)
    
    metric(eq_odds_group_0_test_model.label, eq_odds_group_0_test_model.pred,
           eq_odds_group_1_test_model.label, eq_odds_group_1_test_model.pred)
    y_cd = cd(eq_odds_group_0_test_model.pred.round(), eq_odds_group_1_test_model.pred.round(),\
       cd_eq_odds_group_0_test_model.pred.round(),cd_eq_odds_group_1_test_model.pred.round())
    
       
    group_0_test_data['pred'] = eq_odds_group_0_test_model.pred.round()
    group_1_test_data['pred'] = eq_odds_group_1_test_model.pred.round()
    df = group_0_test_data.append(group_1_test_data)
    df = df.drop(['group', 'prediction', 'label'], axis=1).sample(frac=1)
    df.to_csv("results_Hardt/german_test_repaired.csv", index=False)
    np.savetxt("results_Hardt/german_test_repaired_cd.csv", y_cd, delimiter=",")
        
def Hardt(dataset):
    
    make_dataset(dataset)
    if dataset == 'adult':
        Adult()
    elif dataset == 'compas':
        Compas()
    elif dataset == 'german':
        German()
    