from __future__ import division
import sys
import matplotlib as mpl
mpl.use('TkAgg')
from Modules.MatrixOprations.contin_table import *
import random
#import tensorly as tl
import numpy as np
from scipy.sparse import csr_matrix, find
from  Core.grounder import  data_from_sat,grounder4,sing_grounder
matplotlib.use('TkAgg')
import multiprocessing
from functools import partial
from contextlib import contextmanager
import timeit
start_time = timeit.default_timer()
import psutil

def chunks(l, n):
    n=int(n)
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))

class Repair(object):
    def __init__(self):
        pass



    def from_file_indep_repair(self, path, X, Y, Z=[], method='MF',k=5,n_parti=8, sample_frac=1,smother=1, insert_ratio=1,in_weight=10000000,out_weight=10000000,conf_weight=10000000):
        rep = Repair()
        run_time=[]
        sat_times=[]
        bef_CMI=[]
        af_CMI=[]
        memoryUses=[]
        size=0
        for j in range(0, k):
            cur_train = read_from_csv(path + str(j) + '.csv')
            #cur_train=cur_train.sample(frac=0.01)
            inf=Info(cur_train)
            CMI=inf.CMI(X, Y, Z)
            bef_CMI.insert(0,CMI)
            #print('Data before repair:',len(cur_train.index),'CMI:',CMI ,'sample_frac: ',sample_frac, 'conf_weight',conf_weight)
            print('Method:', method)
            start_time = timeit.default_timer()
            size=len(cur_train.index)
            rep_train = rep.indep_repair(cur_train, path + str(j), X, Y, Z, method=method,insert_ratio=insert_ratio, chunk_size=n_parti, sample_size=sample_frac,  conf_weight=conf_weight)
            pid = os.getpid()
            py = psutil.Process(pid)
            memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
            #print('memory use:', memoryUse)
            end=timeit.default_timer() - start_time
            run_time.insert(0,end)
            inf = Info(rep_train)
            CMI=inf.CMI(X, Y, Z)
            af_CMI.insert(0,CMI)
            memoryUses.insert(0,memoryUse)
            #print('Data after repair', len(rep_train.index),'CMI:', inf.CMI(X,Y,Z), 'time: ',end)

            if method!='sat':
              rep_train.to_csv(path+'_rep'+method+'_'+ str(j) + '.csv', encoding='utf-8', index=False)
              #print(path + '_rep' + method + '_' + str(j) + '.csv')
            else:
                if 0<1:
                    path1=path + '_rep' + str(method) + '_' + str(insert_ratio)+  '_' + str(j)+'.csv'
                else:
                    path1=str(path) + '_rep' + str(method) + '_' + str(j)+'.csv'
                rep_train.to_csv(path1,index=False,encoding='utf-8')
                #print(path1)


            j += 1
        return run_time, sat_times,bef_CMI,af_CMI,memoryUses,size

    @contextmanager
    def poolcontext(*args, **kwargs):
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()

    def  tensor_repair(self,data,D_features, Y_features,Z_features=[]):

        columns=data.columns
        X_features=[]
        for att in columns:
            if att not in D_features and att not in Y_features:
                X_features.insert(0,att)

        X_features=X_features
        res = data.groupby(X_features)

        z = get_distinct(data, X_features)

        tbl_XY = ContinTable()
        tbl_XY.data_to_cnt(data, D_features, Y_features)

        i = len(tbl_XY.col_index)
        j = len(tbl_XY.col_index)

        tensor = tl.tensor(np.arange(i * j * z).reshape((z, i, j)))
        # print(tensor)
        slice = np.arange(i * j).reshape(i, j)
        print(tensor[0, :, :])
        cnt = 0
        slice.fill(0)
        Z_index=dict()
        for name, group in res:
            tbl = ContinTable()
            tbl.data_to_cnt(group, D_features, Y_features)
            # print(name)
            print(tbl.table)




            for col in tbl.col_index:
                for row in tbl.row_index:
                    print(tbl.table.get_value(row, col))
                    if tbl.table.get_value(row, col):
                        r_inx = tbl_XY.row_index.index(row)
                        c_inx = tbl_XY.col_index.index(col)
                        val = tbl.table.get_value(row, col)
                        if tbl.table.shape == (2, 2):
                         slice[r_inx, c_inx] = val
                        else:
                            slice[r_inx, c_inx] = val
            print(slice)
            tensor[cnt, :, :] = slice
            cnt += 1
            print(name)
            Z_index[cnt]=name
        print(tensor)
        #core, factors = tucker(tensor, rank=[z, 1, 1])
        # Reconstruct the full tensor from the decomposed form

        # print(factors)
        #res = tl.tucker_to_tensor(core, factors)
        # print(res)

        X, Y, Z = non_negative_parafac(tensor, rank=2)

        print('X: \n', X)
        print('Y: \n', Y)
        print('Z:\n', Z)
        factors = parafac(tensor, rank=2, init='random', tol=10e-6)
        print(factors)
        rep_tensor = tl.kruskal_to_tensor(factors)

        #print(rep_tensor)

        #rep_tensor = np.round(np.einsum('ir, jr, kr -> ijk', X, Y, Z))
        #print(rep_tensor)

        #core, tucker_factors = tucker(tensor, ranks=[2, 1, 1], init='random', tol=10e-5)
        #for fac in tucker_factors:
        #    print(core.shape,fac.shape)
        #rep_tensor = tl.tucker_to_tensor(core, tucker_factors)
        #print(tucker_factors)

        df = pd.DataFrame(columns=D_features+Y_features+X_features)

        i = len(tbl_XY.col_index)
        j = len(tbl_XY.col_index)
        total=0
        for cnt in range(0,z):
            print('Z',cnt)
            slice=rep_tensor[cnt,:,:]
            print('org \n',tensor[cnt, :, :])
            print('rep \n',slice)
            for row in range(0,i):
                for col in range(0,j):
                    freq=slice.item(row,col)
                    freq = int(freq) + (random.random() < freq - int(freq))
                    total += freq
                    print(freq)
                    for itr in range(0, freq):
                        #print('################')
                        #print('org \n', tensor[cnt, :, :])
                        #print('rep1 \n', slice)
                        #print('oldrep \n', rep_tensor[cnt, :, :])
                        #print(freq)
                        #print([tbl_XY.row_index[row]])
                        #print(list(Z_index[cnt+1]))
                        #print([tbl_XY.col_index[col]])
                        df.loc[-1] = [tbl_XY.row_index[row]]+[tbl_XY.col_index[col]]+list(Z_index[cnt+1])
                        df.index = df.index + 1  # shifting index
                        df = df.sort_index()
                        #print(df.index)


        Contin_XY=ContinTable()
        Contin_XY.data_to_cnt(data,D_features,Y_features)
        #print('Original: \n', len(data.index),Contin_XY.matrix)
        #print('MI: \n',Contin_XY.mi())

        Contin_XY=ContinTable()
        Contin_XY.data_to_cnt(df,D_features,Y_features)
        #print('Rep: \n', len(df.index),Contin_XY.matrix)
        print('MI: \n',Contin_XY.mi())
        #print(total)
        return df

    def repair(self, data,path, X, Y, Z=[], method='MF',insert_ratio=1,sample_size=1, chunk_size=1,conf_weight=1):
        absFilePath = os.path.abspath(__file__)
        fileDir = os.path.dirname(os.path.abspath(__file__))
        directory = os.path.dirname(fileDir)
        if method=='sat' or  method=='naive':
            prov_not_in_by_idnt, prov_in_by_idnt, naive_rep, out,features = sing_grounder(data, X, Y, Z, insert_ratio=insert_ratio, path=path,
                                                                       sample_size=sample_size, conf_weight=conf_weight)
            #print(prov_not_in_by_idnt)
            #print(prov_in_by_idnt)
            #print(prov_not_in_by_idnt, prov_in_by_idnt)
            if method == 'sat':
                #from pysat.examples.fm import FM
                #from pysat.formula import WCNF
                #wcnf = WCNF(from_file=out)
                #fm = FM(wcnf, verbose=0)
                #fm.compute()
                #print(fm.cost)
                #solution=list(fm.model)

                os.system(
                        fileDir+'/SatSolvers/open-wbo-master/./open-wbo  ' + out + ' >>' + out+'sol')  #-algorithm=5
            try:
                df = data_from_sat(features, prov_not_in_by_idnt, prov_in_by_idnt,naive_rep, out+'sol',path,insert_ratio,chunk_size,method=method,solution=[],sol=False)
                return df
            except ValueError:
                  return pd.DataFrame()
        else:
            if Z != []:
                grouped = data.groupby(Z)
                i = 0
                for name, group in grouped:
                    df = self.marginal_repair(group, X, Y,method=method)
                    z=0
                    if type(name)!=tuple:
                        name = [name]
                    if len(name)==0:
                        name=[name]
                    for col in Z:
                         df.insert(0,col,name[z])
                         z+=1
                    j=0
                    for att in Z:
                        if att not in df.columns:
                            df.insert(0, att, name[j])
                            j+=1
                    # print(df.columns)
                    if i > 0:
                        df0 = df0.append(df)
                    else:
                        df0 = df
                        i += 1


            else:
                df0 = self.marginal_repair(data, X, Y, method=method)
            return df0




    def indep_repair(self, data, path, X, Y, Z=[], method='MF', sample_size=1,insert_ratio=1, chunk_size=1, smother=1, conf_weight=1):
        # partitions the data into chunks
        data['W']=1
        if Z==[]:
            Z=['W']
        if method in ['sat','MF','IC','naive','kl']:
            if chunk_size>-1:
                groups=data.groupby(Z)
                groups_size=len(groups)
                dfs = [0] * chunk_size
                sizes = [0] * chunk_size
                i=0
                for name, group in groups:

                    index=sizes.index(min(sizes))
                    if sizes[index]==0:
                        dfs[index]=group
                        sizes[index]=len(group.index)
                    else:
                        df=pd.concat([dfs[index],group])
                        dfs[index]=df
                        sizes[index] = len(dfs[index])
                    i=i+1

            dfs=dfs[0:i]
            sizes = sizes[0:i]
            #print('##########################',chunk_size, len(data.index))
            #print(sizes)
            large=dfs
            @contextmanager
            def poolcontext(*args, **kwargs):
                pool = multiprocessing.Pool(*args, **kwargs)
                yield pool
                pool.terminate()

            #print('#######################################',len(dfs))
            with poolcontext(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(partial(self.repair, path=path, X=X, Y=Y, Z=Z, method=method,insert_ratio=insert_ratio, sample_size=sample_size,conf_weight=conf_weight,chunk_size=chunk_size), large)


            pool.close()
            pool.join()
            return pd.concat(results)




    def old2_marginal_repair(self, data, X, Y, method='MF'):
        self.data = data  # A continacy table as Matrix
        self.columns = data.columns
        self.X = X
        self.Y = Y
        rest = []
        for att in self.columns:
            if att not in X and att not in Y and att:
                rest.insert(0, att)

        # computes the continacy table of X and Y and repair it

        Contin_XY = ContinTable()
        Contin_XY.data_to_cnt(data, X, Y)
        print('Original: \n', Contin_XY.matrix)
        print('Coupling: \n', Contin_XY.indep_cop())
        print('MI: \n', Contin_XY.mi())

        shuffle_table = ContinTable()
        data_shuff = data.copy()
        mi = 100

        '''
        while mi>0.0001:
            data_shuff[Y] = data[Y].transform(np.random.permutation)
            shuffle_table.data_to_cnt(data_shuff,X,Y)
            print('shuff_Table',shuffle_table.matrix)
            mi=shuffle_table.mi()
            print('MI Shuffle: \n', mi)
            print('Low rank',Contin_XY.low_rank_approx())

        contin_matrix_XY_REP=shuffle_table.matrix
        '''
        data_shuff[Y] = data[Y].transform(np.random.permutation)
        shuffle_table.data_to_cnt(data_shuff, X, Y)
        print('shuff_Table', shuffle_table.matrix)

        if method == 'MF':
            #print('SVD')
            contin_matrix_XY_REP = Contin_XY.low_rank_approx()  ## repaied contingacy table

        elif method == 'IC':
            print('method=coupling')
            contin_matrix_XY_REP = Contin_XY.indep_cop()

        # print(Contin_XY.row_index)
        # print(Contin_XY.col_index)
        Contin_XY.matrix = contin_matrix_XY_REP
        Contin_XY_REP = ContinTable()
        Contin_XY_REP.matrix_to_cnt(contin_matrix_XY_REP)
        print('Repaired: \n', Contin_XY_REP.matrix)
        print('MI: \n', Contin_XY_REP.mi())

        '''
        if Contin_XY_REP.matrix.shape==(2,2):
            c1=Contin_XY_REP.matrix.item(0,0)+Contin_XY_REP.matrix.item(0,1)
            c2 = Contin_XY_REP.matrix.item(1, 0) + Contin_XY_REP.matrix.item(1, 1)
            rep=(c1*Contin_XY_REP.matrix.item(0,1)/c2)-Contin_XY_REP.matrix.item(0, 1)
            Contin_XY_REP.matrix[0, 1]=int(rep)+ Contin_XY_REP.matrix[0, 1]
        print('repared \n',Contin_XY_REP.matrix)
        # computes conditional probabalities of P(Z|XY)
        '''

        Contin_Full = ContinTable()
        Contin_Full.data_to_cnt(data, X + Y, rest)
        dim = Contin_Full.matrix.shape
        w, h = dim[1], dim[0];
        x = Contin_XY.matrix.shape[0] * Contin_XY.matrix.shape[1]
        y = dim[0]
        if y != x:
            return pd.DataFrame(columns=X + Y + rest)
        if dim[1] < 2 or dim[0] < 2:
            return pd.DataFrame(columns=X + Y + rest)
        con_prop = [[0.0 for x in range(w)] for y in range(h)]
        con_prop = asmatrix(con_prop)
        ct_dim = Contin_XY.matrix.shape
        devisor = ct_dim[1]
        for i in range(0, dim[0]):
            e = Contin_Full.matrix[i, :]
            mrg = sum(e)
            # (i)
            # print(int(i / devisor), i % devisor)
            for j in range(0, dim[1]):
                el1 = Contin_Full.matrix.item(i, j)
                # el=el1/float(mrg)
                freq = Contin_XY.matrix.item(int(i / devisor), i % devisor)
                a = random.randint(0, 10)
                con_prop[i, j] = el1 * freq / float(mrg)
                # if a<=4:
                #  con_prop[i, j]=math.ceil(el*freq)
                #  print(el*freq)
                #  print(con_prop[i, j])
                #  #print(con_prop[i, j], freq, el1)
                # else:
                #  con_prop[i, j] = math.ceil(el * freq)
                #  #print(con_prop[i, j],freq,el1)

        ### repair
        for i in range(0, dim[0]):
            A = con_prop[i, :]
            # print(A, type(A))
            dim = A.shape
            A = np.array(A).flatten().tolist()
            freq = Contin_XY.matrix.item(int(i / devisor), i % devisor)
            B = [int(e) + (random.random() < e - int(e)) for e in A]
            #print(sum(B), freq)

            # rounded_A=smart_round(A,int(freq))

            # The optimal Lagrange multiplier for a constraint
            # is stoanswerred in constraint.dual_value.
            con_prop[i, :] = B

        # con_prop=con_prop.astype(int)
        final_table = ContinTable()
        final_table.matrix_to_cnt(con_prop)
        # final_table.matrix=final_table.matrix.astype(int)
        # print(final_table.matrix.sum())
        # print(Contin_Full.matrix.sum())
        # print(final_table.Xmrn)
        # print(Contin_XY_REP)
        # final_table.Xmrn.reshape(Contin_XY.matrix.shape)
        print(final_table.Xmrn, Contin_XY.matrix.shape)
        m = np.reshape(final_table.Xmrn, Contin_XY.matrix.shape)

        tbl3 = ContinTable()
        tbl3.matrix_to_cnt(m)
        # print(tbl3.mi(m))
        # print(Contin_XY.mi())
        # print(final_table.Xmrn)

        Contin_Full.matrix = final_table.matrix
        # print(Contin_Full.matrix,Contin_Full.col_index,Contin_Full.row_index)
        Contin_Full.matrix = np.asarray(Contin_Full.matrix.astype(int))
        # print(Contin_Full.matrix)
        # print(Contin_Full.row_index,Contin_Full.col_index)
        # print(Contin_Full.matrix)

        df = pd.DataFrame(columns=X + Y + rest)
        j = 0
        for row in Contin_Full.row_index:
            i = 0
            for col in Contin_Full.col_index:
                # print([Contin_Full.matrix.item(j, i)] + list(row) + list(col))
                for z in range(0, Contin_Full.matrix.item(j, i)):
                    df.loc[-1] = list(row) + list(col)
                    df.index = df.index + 1  # shifting index
                    df = df.sort_index()
                i += 1
            j += 1
        Contin_XY = ContinTable()
        Contin_XY.data_to_cnt(df, X, Y)
        print('Original: \n', Contin_XY.matrix)
        print('MI: \n', Contin_XY.mi())
        #   df[Y[0]] = pd.to_numeric(df[Y[0]])
        #   ate = df.groupby(X)[Y[0]].mean()
        #   print(ate)
        return df

    def old_marginal_repair(self, data, X, Y, method='MF'):
        self.data = data  # A continacy table as Matrix
        self.columns = data.columns
        self.X = X
        self.Y = Y
        rest = []
        for att in self.columns:
            if att not in X and att not in Y and att:
                rest.insert(0, att)

        # computes the continacy table of X and Y and repair it

        Contin_XY = ContinTable()
        Contin_XY.data_to_cnt(data, X, Y)
        print('Original: \n', Contin_XY.matrix)
        print('Coupling: \n', Contin_XY.indep_cop())
        print('MI: \n', Contin_XY.mi())

        #shuffle_table = ContinTable()
        #data_shuff = data.copy()
        #mi = 100

        '''
        while mi>0.0001:
            data_shuff[Y] = data[Y].transform(np.random.permutation)
            shuffle_table.data_to_cnt(data_shuff,X,Y)
            print('shuff_Table',shuffle_table.matrix)
            mi=shuffle_table.mi()
            print('MI Shuffle: \n', mi)
            print('Low rank',Contin_XY.low_rank_approx())

        contin_matrix_XY_REP=shuffle_table.matrix
        '''
        #data_shuff[Y] = data[Y].transform(np.random.permutation)
        #shuffle_table.data_to_cnt(data_shuff, X, Y)
        #print('shuff_Table', shuffle_table.matrix)

        if method == 'MF':
            #print('SVD')
            contin_matrix_XY_REP = Contin_XY.low_rank_approx()  ## repaied contingacy table

        elif method == 'IC':
            #print('method=coupling')
            contin_matrix_XY_REP = Contin_XY.indep_cop()

        # print(Contin_XY.row_index)
        # print(Contin_XY.col_index)
        Contin_XY.matrix = contin_matrix_XY_REP
        Contin_XY_REP = ContinTable()
        Contin_XY_REP.matrix_to_cnt(contin_matrix_XY_REP)
        print('Repaired: \n', Contin_XY_REP.matrix)
        print('MI: \n', Contin_XY_REP.mi())

        '''
        if Contin_XY_REP.matrix.shape==(2,2):
            c1=Contin_XY_REP.matrix.item(0,0)+Contin_XY_REP.matrix.item(0,1)
            c2 = Contin_XY_REP.matrix.item(1, 0) + Contin_XY_REP.matrix.item(1, 1)
            rep=(c1*Contin_XY_REP.matrix.item(0,1)/c2)-Contin_XY_REP.matrix.item(0, 1)
            Contin_XY_REP.matrix[0, 1]=int(rep)+ Contin_XY_REP.matrix[0, 1]
        print('repared \n',Contin_XY_REP.matrix)
        # computes conditional probabalities of P(Z|XY)
        '''

        Contin_Full = ContinTable()
        Contin_Full.data_to_cnt(data, X + Y, rest)
        dim = Contin_Full.matrix.shape
        w, h = dim[1], dim[0];
        x = Contin_XY.matrix.shape[0] * Contin_XY.matrix.shape[1]
        y = dim[0]
        if y != x:
            return pd.DataFrame(columns=X + Y + rest)
        if dim[1] < 2 or dim[0] < 2:
            return pd.DataFrame(columns=X + Y + rest)
        con_prop = [[0.0 for x in range(w)] for y in range(h)]
        con_prop = asmatrix(con_prop)
        ct_dim = Contin_XY.matrix.shape
        devisor = ct_dim[1]
        for i in range(0, dim[0]):
            e = Contin_Full.matrix[i, :]
            mrg = sum(e)
            # (i)
            # print(int(i / devisor), i % devisor)
            for j in range(0, dim[1]):
                el1 = Contin_Full.matrix.item(i, j)
                # el=el1/float(mrg)
                freq = Contin_XY.matrix.item(int(i / devisor), i % devisor)
                a = random.randint(0, 10)
                con_prop[i, j] = el1 * freq / float(mrg)
                # if a<=4:
                #  con_prop[i, j]=math.ceil(el*freq)
                #  print(el*freq)
                #  print(con_prop[i, j])
                #  #print(con_prop[i, j], freq, el1)
                # else:
                #  con_prop[i, j] = math.ceil(el * freq)
                #  #print(con_prop[i, j],freq,el1)

        ### repair
        for i in range(0, dim[0]):
            A = con_prop[i, :]
            # print(A, type(A))
            dim = A.shape
            A = np.array(A).flatten().tolist()
            freq = Contin_XY.matrix.item(int(i / devisor), i % devisor)
            B = [int(e) + (random.random() < e - int(e)) for e in A]
            print(sum(B), freq)

            # rounded_A=smart_round(A,int(freq))

            # The optimal Lagrange multiplier for a constraint
            # is stoanswerred in constraint.dual_value.
            con_prop[i, :] = B

        # con_prop=con_prop.astype(int)
        final_table = ContinTable()
        final_table.matrix_to_cnt(con_prop)
        # final_table.matrix=final_table.matrix.astype(int)
        # print(final_table.matrix.sum())
        # print(Contin_Full.matrix.sum())
        # print(final_table.Xmrn)
        # print(Contin_XY_REP)
        # final_table.Xmrn.reshape(Contin_XY.matrix.shape)
        print(final_table.Xmrn, Contin_XY.matrix.shape)
        m = np.reshape(final_table.Xmrn, Contin_XY.matrix.shape)

        tbl3 = ContinTable()
        tbl3.matrix_to_cnt(m)
        # print(tbl3.mi(m))
        # print(Contin_XY.mi())
        # print(final_table.Xmrn)

        Contin_Full.matrix = final_table.matrix
        # print(Contin_Full.matrix,Contin_Full.col_index,Contin_Full.row_index)
        Contin_Full.matrix = np.asarray(Contin_Full.matrix.astype(int))
        # print(Contin_Full.matrix)
        # print(Contin_Full.row_index,Contin_Full.col_index)
        # print(Contin_Full.matrix)

        df = pd.DataFrame(columns=X + Y + rest)
        j = 0
        for row in Contin_Full.row_index:
            i = 0
            for col in Contin_Full.col_index:
                # print([Contin_Full.matrix.item(j, i)] + list(row) + list(col))
                for z in range(0, Contin_Full.matrix.item(j, i)):
                    df.loc[-1] = list(row) + list(col)
                    df.index = df.index + 1  # shifting index
                    df = df.sort_index()
                i += 1
            j += 1
        Contin_XY = ContinTable()
        Contin_XY.data_to_cnt(df, X, Y)
        print('Original: \n', Contin_XY.matrix)
        print('MI: \n', Contin_XY.mi())
        #   df[Y[0]] = pd.to_numeric(df[Y[0]])
        #   ate = df.groupby(X)[Y[0]].mean()
        #   print(ate)
        return df

    def marginal_repair(self, data, X, Y, method='MF'):
        print("marginal_repair")
        self.data = data  # A continacy table as Matrix
        self.columns = data.columns
        self.X = X
        self.Y = Y
        rest = []
        for att in self.columns:
            if att not in X and att not in Y and att:
                rest.insert(0, att)

        # computes the continacy table of X and Y and repair it

        Contin_XY = ContinTable()
        Contin_XY.data_to_cnt(data, X, Y)
        #print('Original: \n', Contin_XY.matrix)
        #print('Coupling: \n', Contin_XY.indep_cop())
        #print('MI: \n', Contin_XY.mi())



        if method == 'MF':
            #print('SVD')
            contin_matrix_XY_REP = Contin_XY.low_rank_approx()  ## repaied contingacy table

        if method == 'kl':
            #print('SVD')
            contin_matrix_XY_REP = Contin_XY.low_rank_approx(loss='kullback-leibler')
        elif method == 'IC':
            #print('method=coupling')
            contin_matrix_XY_REP = Contin_XY.indep_cop()

        print(Contin_XY.row_index)
        print(Contin_XY.col_index)
        print(Contin_XY.matrix)
        print(contin_matrix_XY_REP)
        #Contin_XY.matrix = contin_matrix_XY_REP
        Contin_XY_REP = ContinTable()
        Contin_XY_REP.matrix_to_cnt(contin_matrix_XY_REP)
        #print('Repaired: \n', Contin_XY_REP.matrix)
        #print('MI: \n', Contin_XY_REP.mi())

        '''
        if Contin_XY_REP.matrix.shape==(2,2):
            c1=Contin_XY_REP.matrix.item(0,0)+Contin_XY_REP.matrix.item(0,1)
            c2 = Contin_XY_REP.matrix.item(1, 0) + Contin_XY_REP.matrix.item(1, 1)
            rep=(c1*Contin_XY_REP.matrix.item(0,1)/c2)-Contin_XY_REP.matrix.item(0, 1)
            Contin_XY_REP.matrix[0, 1]=int(rep)+ Contin_XY_REP.matrix[0, 1]
        print('repared \n',Contin_XY_REP.matrix)
        # computes conditional probabalities of P(Z|XY)
        '''
        #dim=Contin_XY_REP.matrix.shape
        #for i in range(0, dim[0]):
        #    for j in range(0, dim[1]):
        #        e = Contin_XY_REP.matrix.item(i, j)
        #        e=int(e) + (random.random() < e - int(e))
        #        Contin_XY_REP.matrix[i, j]=e


        #Contin_XY_REP = ContinTable()
        #Contin_XY_REP.matrix_to_cnt(contin_matrix_XY_REP)
        #print('Repaired: \n', Contin_XY_REP.matrix)
        #print('MI: \n', Contin_XY_REP.mi())


        # a contingacy table over all attributes of dataset
        Contin_Full = ContinTable()
        Contin_Full.data_to_cnt(data, X ,Y)
        Contin_Full.matrix=contin_matrix_XY_REP


        final_table = ContinTable()
        final_table.matrix_to_cnt(Contin_Full.matrix)
        # final_table.matrix=final_table.matrix.astype(int)
        # print(final_table.matrix.sum())
        # print(Contin_Full.matrix.sum())
        # print(final_table.Xmrn)
        # print(Contin_XY_REP)
        # final_table.Xmrn.reshape(Contin_XY.matrix.shape)
        # print(tbl3.mi(m))
        # print(Contin_XY.mi())
        # print(final_table.Xmrn)

        Contin_Full.matrix = final_table.matrix
        # print(Contin_Full.matrix,Contin_Full.col_index,Contin_Full.row_index)
        Contin_Full.matrix = np.asarray(Contin_Full.matrix.astype(int))
        # print(Contin_Full.matrix)
        # print(Contin_Full.row_index,Contin_Full.col_index)
        # print(Contin_Full.matrix)

        df = pd.DataFrame(columns=X + Y)
        j = 0
        for row in Contin_Full.row_index:
            i = 0
            for col in Contin_Full.col_index:
                # print([Contin_Full.matrix.item(j, i)] + list(row) + list(col))
                for z in range(0, Contin_Full.matrix.item(j, i)):
                    if type(row)!=tuple:
                        row=tuple([row])
                    if type(col)!=tuple:
                        col=tuple([col])
                    df.loc[-1] = row + col
                    df.index = df.index + 1  # shifting index
                    df = df.sort_index()
                i += 1
            j += 1
        Contin_XY = ContinTable()
        Contin_XY.data_to_cnt(df, X, Y)
        #print('Original: \n', Contin_XY.matrix)
        #print('MI: \n', Contin_XY.mi())
        #   df[Y[0]] = pd.to_numeric(df[Y[0]])
        #   ate = df.groupby(X)[Y[0]].mean()
        #   print(ate)
        return df

    def new_marginal_repair(self, data, X, Y, method='MF'):
        self.data = data  # A continacy table as Matrix
        self.columns = data.columns
        self.X = X
        self.Y = Y
        rest = []
        for att in self.columns:
            if att not in X and att not in Y and att:
                rest.insert(0, att)

        # computes the continacy table of X and Y and repair it

        Contin_XY = ContinTable()
        Contin_XY.data_to_cnt(data, X, Y)
        print('Original: \n', Contin_XY.matrix)
        #print('Coupling: \n', Contin_XY.indep_cop())
        print('MI: \n', Contin_XY.mi())



        if method == 'MF':
            print('SVD')
            contin_matrix_XY_REP = Contin_XY.low_rank_approx()  ## repaied contingacy table

        elif method == 'IC':
            print('method=coupling')
            contin_matrix_XY_REP = Contin_XY.indep_cop()

        # print(Contin_XY.row_index)
        # print(Contin_XY.col_index)
        #Contin_XY.matrix = contin_matrix_XY_REP
        Contin_XY_REP = ContinTable()
        Contin_XY_REP.matrix_to_cnt(contin_matrix_XY_REP)
        print('Repaired: \n', Contin_XY_REP.matrix)
        print('MI: \n', Contin_XY_REP.mi())

        '''
        if Contin_XY_REP.matrix.shape==(2,2):
            c1=Contin_XY_REP.matrix.item(0,0)+Contin_XY_REP.matrix.item(0,1)
            c2 = Contin_XY_REP.matrix.item(1, 0) + Contin_XY_REP.matrix.item(1, 1)
            rep=(c1*Contin_XY_REP.matrix.item(0,1)/c2)-Contin_XY_REP.matrix.item(0, 1)
            Contin_XY_REP.matrix[0, 1]=int(rep)+ Contin_XY_REP.matrix[0, 1]
        print('repared \n',Contin_XY_REP.matrix)
        # computes conditional probabalities of P(Z|XY)
        '''
        #dim=Contin_XY_REP.matrix.shape
        #for i in range(0, dim[0]):
        #    for j in range(0, dim[1]):
        #        e = Contin_XY_REP.matrix.item(i, j)
        #        e=int(e) + (random.random() < e - int(e))
        #        Contin_XY_REP.matrix[i, j]=e


        #Contin_XY_REP = ContinTable()
        #Contin_XY_REP.matrix_to_cnt(contin_matrix_XY_REP)
        #print('Repaired: \n', Contin_XY_REP.matrix)
        #print('MI: \n', Contin_XY_REP.mi())


        # a contingacy table over all attributes of dataset
        Contin_Full = ContinTable()
        Contin_Full.data_to_cnt(data, X + Y, rest)
        dim = Contin_Full.matrix.shape

        org=Contin_XY.matrix.flatten()
        org=org.tolist()
        if  type(org[0])==list:
            org = org[0]
        rep=contin_matrix_XY_REP.flatten()
        rep=rep.tolist()
        if  type(rep[0])==list:
            rep = rep[0]


        print(Contin_XY.row_index)
        print(Contin_XY.col_index)

        print(Contin_Full.row_index)
        print(Contin_Full.col_index)
        for i in range(0, dim[0]):
            for j in range(0, dim[1]):
                if org[i]>0:
                     el=Contin_Full.matrix.item(i, j)
                     el1 = (el*rep[i])/(org[i])
                else:
                    el = Contin_Full.matrix.item(i, j)
                    el1 = (el * rep[i])
                el1=int(el1) + (random.random() < el1 - int(el1))
                # el=el1/float(mrg)
                Contin_Full.matrix[i, j]=el1

                # if a<=4:
                #  con_prop[i, j]=math.ceil(el*freq)
                #  print(el*freq)
                #  print(con_prop[i, j])
                #  #print(con_prop[i, j], freq, el1)
                # else:
                #  con_prop[i, j] = math.ceil(el * freq)
                #  #print(con_prop[i, j],freq,el1)

        ### repair
        print('**********************')
        #print(Contin_Full.matrix)

        final_table = ContinTable()
        final_table.matrix_to_cnt(Contin_Full.matrix)
        # final_table.matrix=final_table.matrix.astype(int)
        # print(final_table.matrix.sum())
        # print(Contin_Full.matrix.sum())
        # print(final_table.Xmrn)
        # print(Contin_XY_REP)
        # final_table.Xmrn.reshape(Contin_XY.matrix.shape)
        # print(tbl3.mi(m))
        # print(Contin_XY.mi())
        # print(final_table.Xmrn)

        Contin_Full.matrix = final_table.matrix
        # print(Contin_Full.matrix,Contin_Full.col_index,Contin_Full.row_index)
        Contin_Full.matrix = np.asarray(Contin_Full.matrix.astype(int))
        # print(Contin_Full.matrix)
        # print(Contin_Full.row_index,Contin_Full.col_index)
        # print(Contin_Full.matrix)

        df = pd.DataFrame(columns=X + Y + rest)
        j = 0
        for row in Contin_Full.row_index:
            i = 0
            for col in Contin_Full.col_index:
                # print([Contin_Full.matrix.item(j, i)] + list(row) + list(col))
                for z in range(0, Contin_Full.matrix.item(j, i)):
                    df.loc[-1] = list(row) + list(col)
                    df.index = df.index + 1  # shifting index
                    df = df.sort_index()
                i += 1
            j += 1
        Contin_XY = ContinTable()
        Contin_XY.data_to_cnt(df, X, Y)
        print('Original: \n', Contin_XY.matrix)
        print('MI: \n', Contin_XY.mi())
        #   df[Y[0]] = pd.to_numeric(df[Y[0]])
        #   ate = df.groupby(X)[Y[0]].mean()
        #   print(ate)
        return df



    def marginal_repair1(self,data,X,Y,method='IC'):

        Z=[]   ## Attributes other than X and Y
        for att in data.columns:
            if att not in X and att not in Y and att:
                Z.insert(0,att)

        # computes the continacy table of X and Y and repairs it

        Contin_XY=ContinTable()
        Contin_XY.data_to_cnt(data,X,Y)
        print('Original: \n', Contin_XY.matrix)
        print('Coupling: \n', Contin_XY.indep_cop())
        print('MI: \n',Contin_XY.mi())


        if method=='MF':
            print('Method: SVD')
            contin_matrix_XY_REP=Contin_XY.low_rank_approx()  ## repaied matrix

        elif method=='IC':
          print('Method:Coupling')
          contin_matrix_XY_REP=Contin_XY.indep_cop()


        #repaired table
        Contin_XY.matrix=contin_matrix_XY_REP
        Contin_XY_REP=ContinTable()
        Contin_XY_REP.matrix_to_cnt(contin_matrix_XY_REP)
        print('Repaired: \n', Contin_XY_REP.matrix)
        print('MI: \n',Contin_XY_REP.mi())



        if Z!=[]:

            Contin_Full = ContinTable()
            Contin_Full.data_to_cnt(data, X+Y, Z)
            dim=Contin_Full.matrix.shape
            w, h = dim[1], dim[0];
            x=Contin_XY.matrix.shape[0]*Contin_XY.matrix.shape[1]
            y=dim[0]
            if y!=x:
                return     pd.DataFrame(columns=X+Y+Z)
            if dim[1]<2 or dim[0]<2:
                return pd.DataFrame(columns=X+Y+Z)

            ## compute P(rest|XY)

            con_prop = [[0.0 for x in range(w)] for y in range(h)]
            con_prop=asmatrix(con_prop)
            ct_dim=Contin_XY.matrix.shape
            devisor=ct_dim[1]
            Contin_Full.matrix = csr_matrix(Contin_Full.matrix)
            margins=dict()
            I, J, V = find(Contin_Full.matrix)
            print(I, J, V)
            for i in range(0,I.size):
                if I[i] not in margins.keys():
                    margins[I[i]]=V[i]
                else:
                    margins[I[i]] += V[i]



            print(margins)
            for i in range(0,dim[0]):
                e = Contin_Full.matrix[i, :]
                mrg = sum(e)
                #(i)
                #print(int(i / devisor), i % devisor)
                for j in range(0, dim[1]):
                  el1=Contin_Full.matrix.item(i, j)
                  #el=el1/float(mrg)
                  freq=Contin_XY.matrix.item(int(i / devisor), i % devisor)
                  a = random.randint(0, 10)
                  con_prop[i, j] = el1 * freq/float(mrg)
                  #if a<=4:
                  #  con_prop[i, j]=math.ceil(el*freq)
                  #  print(el*freq)
                  #  print(con_prop[i, j])
                  #  #print(con_prop[i, j], freq, el1)
                  #else:
                  #  con_prop[i, j] = math.ceil(el * freq)
                  #  #print(con_prop[i, j],freq,el1)

        ### couple the distribution on the repaired table with the conditional distribution P(rest|XY)
            for i in range(0,dim[0]):
                A = con_prop[i,:]
                #print(A, type(A))
                dim=A.shape
                A = np.array(A).flatten().tolist()
                freq = Contin_XY.matrix.item(int(i / devisor), i % devisor)
                B=[int(e) + (random.random() < e - int(e)) for e in A]
                print(sum(B),freq)
                con_prop[i,:]=B


        #con_prop=con_prop.astype(int)
        final_table=ContinTable()
        final_table.matrix_to_cnt(con_prop)
        #final_table.matrix=final_table.matrix.astype(int)
        #print(final_table.matrix.sum())
        #print(Contin_Full.matrix.sum())
        #print(final_table.Xmrn)
        #print(Contin_XY_REP)
        #final_table.Xmrn.reshape(Contin_XY.matrix.shape)
        print(final_table.Xmrn,Contin_XY.matrix.shape)
        m=np.reshape(final_table.Xmrn,Contin_XY.matrix.shape)

        tbl3=ContinTable()
        tbl3.matrix_to_cnt(m)
        #print(tbl3.mi(m))
        #print(Contin_XY.mi())
        #print(final_table.Xmrn)

        Contin_Full.matrix=final_table.matrix
        #print(Contin_Full.matrix,Contin_Full.col_index,Contin_Full.row_index)
        Contin_Full.matrix=np.asarray(Contin_Full.matrix.astype(int))
        #print(Contin_Full.matrix)
        #print(Contin_Full.row_index,Contin_Full.col_index)
        #print(Contin_Full.matrix)

        df = pd.DataFrame(columns=X+Y+Z)
        j = 0
        for row in Contin_Full.row_index:
            i = 0
            for col in Contin_Full.col_index:
                #print([Contin_Full.matrix.item(j, i)] + list(row) + list(col))
                for z in range(0, Contin_Full.matrix.item(j, i)):
                    df.loc[-1] = list(row) + list(col)
                    df.index = df.index + 1  # shifting index
                    df = df.sort_index()
                i += 1
            j += 1
        Contin_XY=ContinTable()
        Contin_XY.data_to_cnt(df,X,Y)
        print('Original: \n', Contin_XY.matrix)
        print('MI: \n',Contin_XY.mi())
     #   df[Y[0]] = pd.to_numeric(df[Y[0]])
     #   ate = df.groupby(X)[Y[0]].mean()
     #   print(ate)
        return df











        #for elem in pd.unique(data[rest].values):
        #    print(elem)
        #for name, group in grouped:
        #    print(name, len(group))







if __name__ == '__main__':
    # data = pd.read_csv("/Users/babakmac/Documents/FairDB/Experiments/Data/german/german.csv")
    # data = pd.read_csv("/Users/babakmac/Documents/FairDB/Experiments/Data/german/german.csv")
    # data = pd.read_csv("/Users/babakmac/Documents/FairDB/Experiments/Data/german/german.csv")
    #data=pd.read_csv('/Users/babakmac/Documents/XDBData/bin_german_credit.csv')
    #SMALL TEST - LUKE
    df=pd.DataFrame(columns=[ 'X', 'Y', 'Z'])
    df.loc[-1] = ['a' , 'a' , 'c' ]
    df.index = df.index + 1
    df.loc[-1] = [ 'a' , 'a' , 'c']
    df.index = df.index + 1
    df.loc[-1] = [ 'a' , 'a' , 'c']
    df.index = df.index + 1
    df.loc[-1] = [ 'a' , 'b' , 'c']
    df.index = df.index + 1
    df.loc[-1] = [    'a' , 'b' , 'c']
    df.index = df.index + 1
    df.loc[-1] = [    'b' , 'a' , 'c']
    df.index = df.index + 1
    df.loc[-1] = [ 'b' , 'a' , 'c']
    df.index = df.index + 1
    df.loc[-1] = [ 'b' , 'b' , 'c']
    df.index = df.index + 1
    df.loc[-1] = [ 'b' , 'a' , 'd']
    print(df)
    rep=Repair()
    dfrep=rep.indep_repair(df, './', ['X'], ['Y'],['Z'], method = 'IC')
    print(dfrep)
    dfrep=rep.indep_repair(df, './', ['X'], ['Y'],['Z'], method = 'naive')
    print(dfrep)
    dfrep=rep.indep_repair(df, './', ['X'], ['Y'],['Z'], method = 'MF')
    print(dfrep)
    dfrep=rep.indep_repair(df, './', ['X'], ['Y'],['Z'], method = 'kl')
    print(dfrep)
    dfrep=rep.indep_repair(df, './', ['X'], ['Y'],['Z'], method = 'sat')
    print(dfrep)
