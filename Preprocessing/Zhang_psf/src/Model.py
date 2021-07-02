from __future__ import print_function, division
import pandas as pd
import networkx as nx
import xml.etree.ElementTree as ET
import random, pickle


def LoadGraph(Graph_path):
    '''
    load graph from xml file , then return a networkx objects
    :param Graph_path:
    :return: an netorkx bidirect graph object
    '''

    def make_dict_from_tree(element_tree):
        def internal_iter(tree, accum):
            if tree is None:
                return accum
            if tree.getchildren():
                accum[tree.tag] = {}
                for each in tree.getchildren():
                    result = internal_iter(each, {})
                    if each.tag in accum[tree.tag]:
                        if not isinstance(accum[tree.tag][each.tag], list):
                            accum[tree.tag][each.tag] = [
                                accum[tree.tag][each.tag]
                            ]
                        accum[tree.tag][each.tag].append(result[each.tag])
                    else:
                        accum[tree.tag].update(result)
            else:
                accum[tree.tag] = tree.text
            return accum

        return internal_iter(element_tree, {})

    g = make_dict_from_tree(ET.fromstringlist(open(Graph_path, 'r').readlines()))
    G = nx.DiGraph()
    for v in g['graph']['variables']['variable']:
        G.add_node(v)

    for e in g['graph']['edges']['edge']:
        s, d = e.split(' --> ')
        G.add_edge(s, d)
    return G


def GoThrough(node, path):
    return True if node in path else False


def Identifiable(pi, npi):
    S_pi = list(set([path[1] for path in pi]))
    S_npi = list(set([path[1] for path in npi]))
    if len(set(S_npi + S_pi)) == len(S_pi + S_npi):
        return True
    else:
        return False


def BuildPathSet(G, Cname, Ename, R_sets):
    '''
    build path set from given redline attributes
    :param G:
    :param C:
    :param E:
    :param R_sets: a set of attribute sets. Each set corresponds to a path ser
    :return:
    '''
    S_set = G.succ[Cname]
    O_set = list(set(G.nodes()) - set(S_set) - {Cname})
    # build a set containing all simple paths from C to E
    PI = [path for path in nx.all_simple_paths(G, Cname, Ename)]
    Pi_Set = []
    for Rs in R_sets:
        pi = []
        npi = []
        for path in PI:
            flag = False
            for R in Rs:
                if GoThrough(R, path) or (R == '-' and path == [Cname, Ename]):
                    flag = True
            if flag:
                pi.append(path)
            else:
                npi.append(path)
        if not Identifiable(pi, npi):
            raise ValueError(' kite structure')

        Pi_Set.append(pi)

    Succ_sets = [list(set([path[1] for path in pi])) for pi in Pi_Set]

    return Pi_Set, Succ_sets, O_set


def LoadData(path):
    return pd.read_csv(path, dtype=str)


def Shuffle(df):
    return random.sample(list(df), len(df))


def ListAdd(List1, Varibale):
    return list(List1) + [Varibale]


def ListSubstract(List1, Varibale):
    return list(set(List1) - {Varibale})


def ConditionalProb(Target, target, Given, given, df):
    if Given.__len__() == 0:
        return df[df[Target] == target].__len__() / df.__len__()
    elif Target in Given:
        Given = Given.difference([Target])
        given = given[Given]
    d = df.copy()
    for i, c in enumerate(Given):
        d = d[d[c] == given[i]]
    return d[d[Target] == target].__len__() / d.__len__() if d.__len__() > 0 else 0.0


def Combine(data=None, V_max_list=[], V_min_list=[], way='max'):
    '''
    return a contingency table
    :param data: raw data
    :param V_max_list: a list of variables which aims to return all possible combinations
    :param V_min_list: a list of variables which aims to return less possible combinations
    :param way:
        'max': return all possible combination, V_max_list cannot be null
        'min': return less possible combination, V_min_list cannot be null
        'mixed': return mixed combination, V_max_list and V_min_list cannot be null
    :return: Pandas DataFrame
    '''
    if way == 'max':
        v_Val = pd.DataFrame({'key': [1]})
        for k in V_max_list:
            vals = pd.unique(data[k].values)
            len = vals.__len__()
            df_temp = pd.DataFrame({'key': [1] * len, k: vals})
            v_Val = pd.merge(v_Val, df_temp, on='key')
        return v_Val[sorted(V_max_list)]
    if way == 'min':
        v_Val = data[V_min_list]
        v_Val = v_Val.drop_duplicates()
        v_Val.index = range(v_Val.__len__())
        return v_Val[sorted(V_min_list)]
    if way == 'mixed':
        try:
            v_Val = data[V_min_list]
            v_Val = v_Val.drop_duplicates()
        except:
            v_Val = pd.DataFrame({'key': [1]})
        v_Val['key'] = [1] * v_Val.__len__()
        for k in V_max_list:
            vals = pd.unique(data[k].values)
            len = vals.__len__()
            df_temp = pd.DataFrame({'key': [1] * len, k: vals})
            v_Val = pd.merge(v_Val, df_temp, on='key')
        return v_Val[sorted(V_max_list + V_min_list)]


def FindConditionalProb(Target, target, Given, given, Dist):
    if Given == []:
        hash = str(target)
    elif Target in Given:
        Given = sorted(Given)
        hash = '-'.join(given[Given])
    else:
        Given = sorted(ListAdd(Given, Target))
        given[Target] = target
        hash = '-'.join(given[Given])
    try:
        prob = Dist[Target].loc[hash]['prob']
    except:
        prob = 0.0
    return prob


def AssignValue(v_set, C_name, C_val):
    v_set_copy = v_set.copy()
    v_set_copy[C_name] = C_val
    return v_set_copy


class Cause():
    def __init__(self, name, act, bas):
        self.name = name
        self.act = act
        self.bas = bas


class Effect():
    def __init__(self, name, pos, neg):
        self.name = name
        self.pos = pos
        self.neg = neg


class Model():
    def __init__(self, dataset, df, subdir):
        # load adult dataset and setting
        dir = '../data/'
        if dataset == 'adult':
            self.graph = LoadGraph(dir + subdir + "Adult.xml")
            self.C = Cause(name=u'sex', act=u'Male', bas=u'Female')
            self.E = Effect(name=u'income', pos=u'1', neg=u'0')
            if df is None:
                self.df = LoadData(dir + 'Adult_bin.csv')
                self.PicklePath = dir + subdir + 'Adult.pickle'
            else:
                self.df = df
                self.df.index = range(df.__len__())
                self.PicklePath = ''
        
        if dataset == 'compas':
            self.graph = LoadGraph(dir + subdir + "Compas.xml")
            self.C = Cause(name=u'Race', act=u'Other', bas=u'African-American')
            self.E = Effect(name=u'two_year_recid', pos=u'1', neg=u'0')
            if df is None:
                self.df = LoadData(dir + 'Compas_bin.csv')
                self.PicklePath = dir + subdir + 'Compas.pickle'
            else:
                self.df = df
                self.df.index = range(df.__len__())
                self.PicklePath = ''
                
        if dataset == 'german':
            self.graph = LoadGraph(dir + subdir + "German.xml")
            self.C = Cause(name=u'Sex', act=u'Male', bas=u'Female')
            self.E = Effect(name=u'credit', pos=u'1', neg=u'0')
            if df is None:
                self.df = LoadData(dir + 'German_bin.csv')
                self.PicklePath = dir + subdir + 'German.pickle'
            else:
                self.df = df
                self.df.index = range(df.__len__())
                self.PicklePath = ''

        self.Dist = self.CalDist()

    def CalDist(self):
        try:
            Dist = pickle.load(open(self.PicklePath, 'r'))
            return Dist
        except:
            Varibales = self.graph.nodes()
            Dist = {}
            for V in Varibales:
                Parents = self.graph.pred[V].keys()
                if V == self.E.name:
                    vals = Combine(data=self.df, V_max_list=ListAdd(Parents, self.E.name), V_min_list=[], way='max')
                else:
                    vals = Combine(data=self.df, way='min', V_min_list=ListAdd(Parents, V))
                Given = vals.columns
                vals['prob'] = vals.apply(
                    lambda l: ConditionalProb(Target=V, target=l[V], Given=Given, given=l, df=self.df.copy()), axis=1)
                vals['hash'] = vals.apply(lambda l: '-'.join(l[Given]), axis=1)
                Dist[V] = vals.set_index('hash')
            try:
                pickle.dump(Dist, open(self.PicklePath, 'w'))
            except:
                pass
            return Dist

    def setRedlineAttrSet(self, R_set):
        self.R_set = R_set
        self.Pi_Sets, self.Succ_sets, self.O_set = BuildPathSet(self.graph, self.C.name, self.E.name, self.R_set)
