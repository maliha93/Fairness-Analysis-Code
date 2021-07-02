

import pandas as pd
from utils.util import *
from utils.read_data import read_from_csv
from Modules.MatrixOprations.lowrank_decoms import *
from Modules.InformationTheory.info_theo import *
from numpy import linalg as LA
from scipy.spatial import distance_matrix

class ContinTable(object):
    #  Contingacy Table Class
    def __init__(self):
        self.matrix = None   # A continacy table as Matrix
        self.dim=None        # Contingaxy table dimentions
        self.Ymrn=None       # Y margin
        self.Xmrn = None     # X margin
        pass


    def data_to_cnt(self, data=None, X=None, Y=None):

    # Input: A panda dataframe and two set of attributes
    # Outpout: Creates a contingacy table and its marginals
             self.table= pd.crosstab([data[att] for att in X],[data[att] for att in Y],margins=False)
             #print(self.table)
             self.matrix=np.asmatrix(self.table.values)
             self.dim=list(self.matrix.shape)
             self.Xmrn, self.Ymrn, self.total = self.get_margins(self.matrix)
             self.col_index=list(self.table.columns)
             self.row_index=list(self.table.index)


    def get_margin(self, matrix=None):
    # Returns a vector consists of the sum of columns
        matrix=np.asmatrix(matrix  )
        return np.asmatrix(np.array([e.sum() for e in matrix]))

    def get_margins(self, matrix=None):
    # Returns margins of a contingacy table
        Xmrn = self.get_margin(matrix)
        Ymrn = self.get_margin(matrix.T)
        total = Xmrn.sum()
        return Xmrn, Ymrn,total

    def matrix_to_cnt(self, matrix=None):
        # Input: A matrix
        # Outpout: Marginals of the matrix seen as a contingacy table
        self.matrix=matrix
        self.Xmrn, self.Ymrn, self.total = self.get_margins(matrix)




    def indep_cop(self):
        #return the indepedant coupling of a contingacy table marginals
        print(self.Xmrn.T,self.Ymrn)
        m=(np.matmul(self.Xmrn.T,self.Ymrn)*1/self.total)
        m=np.asmatrix(m)
        return m

    def ent(self, matrix=None,Xmrn=[], Ymrn=[], total=0,dim=0):
        if  not Xmrn.any():
         Xmrn, Ymrn, total = self.get_margins(matrix)
        ent=0
        if dim==0:
            m=Xmrn
        elif dim==1:
            m = Ymrn
        elif dim==2:
            m = np.asmatrix(matrix)
        dim=m.shape
        for i in range(0, dim[0]):
            for j in range(0, dim[1]):
                e=m.item(i,j)
                if e >0:
                    ent=ent-(e/total)*np.log(e/total)

        return ent

    def low_rank_approx(self,  rank=1,loss='frobenius'):
        return  low_rank_approx(self.matrix, rank,loss=loss)

    def mi(self, matrix=[], normal=True):
       #compute mutual information from acontingacy table
        if  matrix==[]:
            matrix=self.matrix
        Xmrn, Ymrn,total = self.get_margins(matrix)
        hx=self.ent(matrix,Xmrn, Ymrn,total,dim=0)
        hy=self.ent(matrix,Xmrn, Ymrn,total,dim=1)
        hxy=self.ent(matrix,Xmrn, Ymrn,total,dim=2)
        mi = hx + hy - hxy
        if normal and hx!=0 and hy!=0:
            mi=mi
        return mi

if __name__ == '__main__':
    data=read_from_csv('/Users/babakmac/Documents/XDBData/binadult2.csv')
    print(data.columns)
    tbl=ContinTable()
    tbl.data_to_cnt(data, 'race', 'income')

    #m=[[2500, 100,1,1,1], [100,2500,1,1,0 ]]
    m=tbl.matrix
    m=np.asmatrix(m)
    tbl.matrix_to_cnt(m)
    print('original: \n',tbl.matrix.astype(int))
    m=tbl.mi(tbl.matrix)
    print('MI',m)
    m=tbl.matrix
    #print(tbl.Xmrn)
    #print(tbl.Ymrn)
    print(tbl.total)


    low_m = low_rank_approx(tbl.matrix, 1)
    print('low rank:\n', low_m.astype(int))
    tbl2 = ContinTable()
    tbl2.matrix_to_cnt(low_m)
    mi2 = tbl2.mi(tbl2.matrix)
    print('MI(low rank)', mi2)
    #print(tbl2.Xmrn)
    #print(tbl2.Ymrn)
    print(tbl2.total)



    indep_m = tbl.indep_cop()
    print('indep:\n', indep_m.astype(int))
    tbl2 = ContinTable()
    tbl2.matrix_to_cnt(indep_m)
    mi3 = tbl2.mi(tbl2.matrix)
    print('MI(indep)', mi3)
    print(tbl2.Xmrn)
    print(tbl2.Ymrn)
    print(tbl2.total)

    print('diff',LA.norm((m-low_m)))
    print('diff', LA.norm((m - indep_m)))
    #inf = Info(data)
    #start = time.time()
    #s1 = inf.CMI(['maritalstatus'],['race'])
    #print('MI',s1)


    ''' 
    tbl.matrix=tbl.indep_cop()
    print('indep:\n',tbl.matrix)
    m=tbl.mutual_info()
    print('MI',m)
    low_m=low_rank_approx(tbl.matrix, 1).astype(int)
    print('low rank:\n',low_m)
    tbl2 = ContinTable()
    m = np.array([[1, 2, 3], [2, 2, 6]])
    tbl2.matrix_to_cnt(low_m)
    m=tbl2.mutual_info()
    print('MI',m)
    '''


