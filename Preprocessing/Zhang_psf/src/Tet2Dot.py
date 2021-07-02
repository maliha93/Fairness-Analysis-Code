from __future__ import print_function
from Model import Model

dir = '../data/'
subdir = 'm0.01/'
RA = 'maritial'

model = Model(dataset='adult', df=None, subdir=subdir)
G = model.graph
R_set = [[RA]]

model.setRedlineAttrSet(R_set=R_set)

edge_set = []
for path in model.Pi_Sets[0]:
    for i in range(path.__len__() - 1):
        edge_set.append((path[i], path[i + 1]))

edges = G.edges()

print('strict digraph  G{')

for edge in edges:
    if edge in edge_set:
        print(edge[0], '->', edge[1], ' ', '[color="blue"];')
    elif edge == (model.C.name, model.E.name):
        print(edge[0], '->', edge[1], ' ', '[color="green"];')
    else:
        print(edge[0], '->', edge[1])

print('}')
