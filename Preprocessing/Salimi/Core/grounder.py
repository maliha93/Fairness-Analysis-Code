
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++  py-sat needs this before python install ..
#
#


from __future__ import division
import matplotlib as mpl
mpl.use('TkAgg')
from Modules.MatrixOprations.contin_table import *
#from Core.indep_repair import Repair
import random
#import tensorly as tl
import numpy as np
#from pysat.examples.fm import FM
#from pysat.formula import WCNF

def list2str(list):
    res=''
    for item in list:
        res+=str(item)+'#'
    return res

def are_eq(a, b):
    return np.array_equal(a,b)

def grounder5(data, repo, X, Y, Z=[]):   ## nested loop
    #data=data[X+Y+Z]
    #data=data[X+Y+Z]
    repos=repo
    prov_not_in_by_value=dict()
    prov_not_in_by_idnt=dict()
    prov_in_by_value=dict()
    prov_in_by_idnt=dict()
    sat_db=dict()
    flashDB(repo)

    grouped=pd.DataFrame({'count' : data.groupby( X+Z+Y ).size()}).reset_index()
    print(grouped)
    df = pd.DataFrame(columns=['k']+X+Z+Y)
    print('Dics cleared')
    #grouped=grouped.reset_index()
    #ate = data.groupby(['origin', 'carrier'])['delayed'].mean()
    #ate.columns = ['Z', 'X', 'Y']
    counter=1
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         print(grouped.columns)
         for j in range(0, int(row[-1])): ##last column contains the frequencies
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value[tmp]=counter
             prov_in_by_idnt[counter]= tmp
             repos.set(counter,tmp)
             print(counter)
             df.index = df.index + 1  # shifting index
             k+=1
             counter+=1
             #df = df.sort_index()
    print('Prov in dic created')
    var_num=counter
    #printDB(prov_in_by_value)
    print(df)
    # K has been inserted to convert a bag to set
    df.reset_index()
    org_joined_df=pd.merge(df,df,left_on=Z, right_on=Z, how='inner')
    joined_df=org_joined_df[['k_x']+ [item+'_x' for item in X]+Z+[item+'_y' for item in Y]].drop_duplicates()
    print('joined_df',len(joined_df.index))
    joined_df.reset_index()

    '''
    counter=0
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         for j in range(0, row[-1]):
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value.set(tmp, counter)
             prov_in_by_idnt.set(counter, tmp)
             counter+=1
    '''

    prov_conter=len(prov_in_by_value.keys())
    for i in range(0,len(joined_df.index)):
         row=joined_df.iloc[i].values
         value=list2str(row)
         tmp=prov_in_by_value.get(value)
         if not tmp:
             prov_conter = prov_conter + 1
             prov_not_in_by_value[value]= prov_conter
             prov_not_in_by_idnt[prov_conter]= value
             repos.set(prov_conter, value)

    not_var_num=prov_conter
    #printDB(prov_not_in_by_value)

    org_joined_df = pd.merge(joined_df, joined_df, left_on=Z, right_on=Z, how='inner') #3370284
    print('joined_df',len(joined_df.index))
    cluse_num=1
    sum_wights=0
    n_in_db=len(prov_in_by_idnt.keys())
    n_not_in_db=len(prov_not_in_by_idnt.keys())


    print('Self joined performed')
    for i in range(1,var_num):
        #w=abs(int(np.random.normal(1000,100)))
        f = random.randint(0, 10)
        if f < 8:
           w= 10000
        else:
           w=1000
        sum_wights+=w
        sat_db[cluse_num]= [w, i,0 ]
        cluse_num += 1
    for i in range(var_num,not_var_num+1):
        #w=abs(int(np.random.normal(5000,100)))
        f = random.randint(0, 10)
        if f < 3:
           w= 10000
        else:
           w=1000
        sum_wights+=w
        sat_db[cluse_num]= [w, -i,0 ]
        cluse_num += 1
    #print(sat_db)
    #tuple_lenght=len(D_features)+len(Y_features)+len(X_features)+1
    #org_joined_df=org_joined_df.drop_duplicates()

    for i in range(1,len(joined_df.index)+1):
        first_row_index=i
        first_tuple=joined_df.iloc[first_row_index-1:first_row_index]
        for j in range(1, len(joined_df.index) + 1):
            sec_row_index = i
            sec_tuple = joined_df.iloc[sec_row_index - 1:sec_row_index]
            if are_eq(first_tuple[Z].values,sec_tuple[Z].values):
                print(i,j)
                k=first_tuple[['k_x']].values
                k1=k[0].tolist()
                x=first_tuple[[item+'_x' for item in X]].values
                x1=x[0].tolist()
                y=first_tuple[[item+'_y' for item in Y]].values
                y1=y[0].tolist()
                z=first_tuple[Z].values
                z1=z[0].tolist()


                k=sec_tuple[['k_x']].values
                k2=k[0].tolist()
                x=sec_tuple[[item+'_x' for item in X]].values
                x2=x[0].tolist()
                y=sec_tuple[[item+'_y' for item in Y]].values
                y2=y[0].tolist()
                z=sec_tuple[Z].values
                z2=z[0].tolist()

                li_first_tuple=list2str(k1+x1+z1+y1)
                li_sec_tuple=list2str(k2+x2+z2+y2)

                first_cols=list2str(k1+x1+z1+y2)
                sec_cols=list2str(k2+x2+z1+y1)

                in_key1 = prov_in_by_value.get(li_first_tuple)
                if not in_key1:
                    in_key1 = prov_not_in_by_value.get(li_first_tuple)

                in_key2 = prov_in_by_value.get(li_sec_tuple)
                if not in_key2:
                    in_key2 = prov_not_in_by_value.get(li_sec_tuple)

                for item in [first_cols,sec_cols]:
                    why_not = prov_in_by_value.get(item)
                    if not why_not:
                        why_not = prov_not_in_by_value.get(item)
                        #         prov_not_in_by_value.set(sec_cols, prov_conter)
                        #        prov_not_in_by_idnt.set(prov_conter, sec_cols)
                        #       why_not=prov_conter
                    if why_not != in_key1 and why_not != in_key2:
                        f = random.randint(0, 10)
                        if f < 9:
                            # pass
                            sat_db[cluse_num] = [sum_wights + 1, in_key1, in_key2, why_not, 0]
                        else:
                            # pass
                            sat_db[cluse_num] = [1000, in_key1, in_key2, why_not, 0]
                        cluse_num += 1


    num_var = n_in_db+not_var_num
    num_cls = len(sat_db.keys())

    file = open("sat.cnf", "w")
    file.write('p' + ' wcnf ' + str(num_var) + ' ' + str(num_cls) + ' '+str(sum_wights+1) + '\n')

    for key,value in sat_db.items():
        value=str(value)
        value=value.replace('[','')
        value=value.replace(']', '')
        value=value.split(',')

        #print(value[0])
        #v1=int(value[0])
        print(len(value))
        if len(value)>3:
         line=str(value[0])+' -'+str(value[1]).replace(' ','')+' -'+str(value[2]).replace(' ','')+' '+str(value[3]).replace(' ','')+' 0 \n'
        else:
         line = str(value[0]) +' ' + str(value[1]).replace(' ', '') +' 0 \n'
        file.write(line)

def data_from_sat(features, prov_not_in_by_idnt, prov_in_by_idnt, naive_rep, out, path, insert_ratio, chunk_size, method,solution,sol=True):
    #prov_not_in_by_value=storage[0]
    #prov_not_in_by_idnt=storage[1]
    #prov_in_by_value=storage[2]
    #prov_in_by_idnt=storage[3]
    #sat_db=storage[4]
    print(out)
    index=[]
    #print(input)
    if sol!=True:
        if method=='sat':
                fileDir = os.path.dirname(os.path.abspath(out))
                with open(out) as file:
                    for line in file.readlines():
                        index=line
                #print(index)
                #print(type(index))
                if(len(index) >= 1):
                    index=index.split(' ')
                    index = index[1:-1]
        else:
                for key in prov_in_by_idnt.keys():
                    if key not in naive_rep:
                        index.insert(0,key)
            #print(index)
            #D_size=len(repos.keys())

    else:
        index=solution
    df = pd.DataFrame(columns=features)
    in_count=0
    out_count=0
    #df.index=1
    #print(prov_in_by_idnt.keys())
    for item in index:
        item=int(item)
        if item>0:
            try:
              value=prov_in_by_idnt.get(item)
              if value is None:
                  value = prov_not_in_by_idnt.get(item)
                  out_count+=1
              else:
                  in_count+=1
              value=value.split('#')
              value = value[0:-1]
              df.loc[-1] = value
              df.index = df.index + 1
              #print(item)
              #print(df.index)

            except AttributeError:
              pass
    org_db_size= len(prov_in_by_idnt.keys())
    print('size of the repaird data:',len(df.index),org_db_size,'number of tuples added:', out_count,'Number of tuples deleted', org_db_size-len(df.index)+out_count)


    try:
        with open(fileDir + '/satinfo.txt', "a") as outfile:
            out=str(insert_ratio)+','+str(len(df.index))+','+str(org_db_size)+','+str(out_count)+','+ str(org_db_size-len(df.index)+out_count)+','+str(chunk_size)+'\n'
            outfile.write(out)
            #print(fileDir + '/satinfo.txt')
    except Exception as e:
     print(e)

    return df


def to_cnf(storage,out):

    prov_not_in_by_value=storage[0]
    prov_not_in_by_idnt=storage[1]
    prov_in_by_value=storage[2]
    prov_in_by_idnt=storage[3]
    sat_db=storage[4]
    prov_not_in_by_idnt.dbsize()

    num_var=prov_in_by_idnt.dbsize()+    prov_not_in_by_idnt.dbsize()
    num_cls=sat_db.dbsize()
    file = open(out, "w")

    file.write('p'+' cnf '+str(num_var)+' '+str(num_cls)+'\n')

    for key in sat_db.scan_iter():
        # delete the key
        # value=r1.delete(key)
        value = sat_db.get(key)




    for key in sat_db.scan_iter():
        value=sat_db.get(key)
        value=value.decode("utf-8")
        value=value.replace('[','')
        value=value.replace(']', '')
        value=value.split(',')

        #print(value[0])
        #v1=int(value[0])
        line='-'+str(value[0])+' -'+str(value[1]).replace(' ','')+' '+str(value[2]).replace(' ','')+' 0 \n'
        file.write(line)

    for key in prov_in_by_value.scan_iter():
        value=prov_in_by_value.get(key)
        print(value)
        value=value.decode("utf-8")
        value=int(value)
        line=str(value)+' 0 \n'
        file.write(line)


def string_to_int(string):
    code = int.from_bytes(str.encode(string), byteorder='big')
    return code

def  int_to_string(int):
    de_code = bytearray.fromhex('{:0192x}'.format(int))
    de_code = de_code.decode("utf-8")
    return  de_code

def grounder3(data, storage, X, Y, Z=[]):
    #data=data[X+Y+Z]
    #data=data[X+Y+Z]

    prov_not_in_by_value=storage[0]
    prov_not_in_by_idnt=storage[1]
    prov_in_by_value=storage[2]
    prov_in_by_idnt=storage[3]
    sat_db=storage[4]
    flashDB(prov_not_in_by_value)
    flashDB(prov_not_in_by_idnt)
    flashDB(prov_in_by_value)
    flashDB(prov_in_by_idnt)
    flashDB(sat_db)
    grouped=pd.DataFrame({'count' : data.groupby( X+Y+Z ).size()}).reset_index()
    df = pd.DataFrame(columns=['k']+X + Y + Z)
    print('Dics cleared')
    #grouped=grouped.reset_index()
    #ate = data.groupby(['origin', 'carrier'])['delayed'].mean()
    #ate.columns = ['Z', 'X', 'Y']
    counter=1
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         for j in range(0, int(row[-1])):
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value.set(tmp, counter)
             prov_in_by_idnt.set(counter, tmp)
             df.index = df.index + 1  # shifting index
             k+=1
             counter+=1
             #df = df.sort_index()
    print('Prov in dic created')
    var_num=counter
    printDB(prov_in_by_value)
    print('joined_df')
    # K has been inserted to convert a bag to set
    df.reset_index()
    joined_df=pd.merge(df,df,left_on=Z, right_on=Z, how='outer')
    joined_df=joined_df[['k_x', X[0]+'_x', Y[0]+'_y']+Z].drop_duplicates()
    print('joined_df')
    joined_df.reset_index()

    '''
    counter=0
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         for j in range(0, row[-1]):
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value.set(tmp, counter)
             prov_in_by_idnt.set(counter, tmp)
             counter+=1
    '''
    sum_wights=0
    cluse_num=1

    prov_conter=prov_in_by_value.dbsize()
    for i in range(0,len(joined_df.index)):
         row=joined_df.iloc[i].values
         value=list2str(row)
         tmp=prov_in_by_value.get(value)
         if not tmp:
             prov_conter = prov_conter + 1
             prov_not_in_by_value.set(value, prov_conter)
             prov_not_in_by_idnt.set(prov_conter, value)
         f = random.randint(0, 10)
         if f < 8:
             w = 10000
         else:
             w = 1000
         sum_wights += w
         sat_db.set(cluse_num, [w, string_to_int(value), 0])
         cluse_num += 1

    not_var_num=prov_conter
    #printDB(prov_not_in_by_value)

    joined_df = pd.merge(joined_df, joined_df, left_on=Z, right_on=Z, how='outer')


    n_in_db=prov_in_by_idnt.dbsize()
    n_not_in_db=prov_not_in_by_idnt.dbsize()
    ratio=n_not_in_db/n_in_db

    print('Self joined performed')
    for i in range(1,var_num):
        #w=abs(int(np.random.normal(1000,100)))
        f = random.randint(0, 10)
        if f < 8:
           w= 10000
        else:
           w=1000
        sum_wights+=w
        sat_db.set(cluse_num, [w, i,0 ])
        cluse_num += 1
    for i in range(var_num,not_var_num+1):
        #w=abs(int(np.random.normal(5000,100)))
        f = random.randint(0, 10)
        if f < 3:
           w= 10000
        else:
           w=1000
        sum_wights+=w
        sat_db.set(cluse_num, [w, -i,0 ])
        cluse_num += 1
    #print(sat_db)


    for i in  range(1, len(joined_df.index)):

        #print(len(joined_df.index))

        row = joined_df.iloc[i].values
        #row=[7 ,'>12' ,0, 'Male', '>=70' ,18 ,'>12' ,1]
        #print(row)
        tmp=list2str(row[3:-3])
        first_cols = list2str(row[0:3])+tmp
        sec_cols = list2str(row[-3:])+tmp
        thris_col=list2str(row[0:2])+list2str([row[-1]])+tmp
        in_key1=string_to_int(first_cols)
        in_key2=string_to_int(sec_cols)
        why_not = string_to_int(thris_col)
        if why_not!=in_key1 and why_not!=in_key2:
            f=random.randint(0,10)
            if f<8:
             pass
             sat_db.set(cluse_num, [sum_wights+1, in_key1, in_key2, why_not, 0])
            else:
             pass
             sat_db.set(cluse_num, [1000, in_key1, in_key2, why_not, 0])
            cluse_num+=1



def grounder4(data, repo, X, Y, Z=[]):
    #data=data[X+Y+Z]
    #data=data[X+Y+Z]
    repos=repo
    prov_not_in_by_value=dict()
    prov_not_in_by_idnt=dict()
    prov_in_by_value=dict()
    prov_in_by_idnt=dict()
    sat_db=dict()
    flashDB(repo)

    grouped=pd.DataFrame({'count' : data.groupby( X+Z+Y ).size()}).reset_index()
    #print(grouped)
    df = pd.DataFrame(columns=['k']+X+Z+Y)
    #print('Dics cleared')
    #grouped=grouped.reset_index()
    #ate = data.groupby(['origin', 'carrier'])['delayed'].mean()
    #ate.columns = ['Z', 'X', 'Y']
    counter=1
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         #print(grouped.columns)
         for j in range(0, int(row[-1])): ##last column contains the frequencies
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value[tmp]=counter
             prov_in_by_idnt[counter]= tmp
             repos.set(counter,tmp)
             #print(counter)
             df.index = df.index + 1  # shifting index
             k+=1
             counter+=1
             #df = df.sort_index()
    #print('Barg is converted to set')
    var_num=counter
    #printDB(prov_in_by_value)
    #print(df)
    # K has been inserted to convert a bag to set
    df.reset_index()
    org_joined_df=pd.merge(df,df,left_on=Z, right_on=Z, how='inner')
    joined_df=org_joined_df[['k_x']+ [item+'_x' for item in X]+Z+[item+'_y' for item in Y]].drop_duplicates()
    #print('joined_df',len(joined_df.index))
    joined_df.reset_index()

    '''
    counter=0
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         for j in range(0, row[-1]):
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value.set(tmp, counter)
             prov_in_by_idnt.set(counter, tmp)
             counter+=1
    '''

    prov_conter=len(prov_in_by_value.keys())
    for i in range(0,len(joined_df.index)):
         row=joined_df.iloc[i].values
         value=list2str(row)
         tmp=prov_in_by_value.get(value)
         if not tmp:
             prov_conter = prov_conter + 1
             prov_not_in_by_value[value]= prov_conter
             prov_not_in_by_idnt[prov_conter]= value
             repos.set(prov_conter, value)

    not_var_num=prov_conter
    #printDB(prov_not_in_by_value)

    org_joined_df = pd.merge(joined_df, joined_df, left_on=Z, right_on=Z, how='inner') #3370284
    print('joined_df',len(joined_df.index))
    cluse_num=1
    sum_wights=0
    n_in_db=len(prov_in_by_idnt.keys())
    n_not_in_db=len(prov_not_in_by_idnt.keys())


    #print('Self joined performed')
    for i in range(1,var_num):
        #w=abs(int(np.random.normal(1000,100)))
        f = random.randint(0, 10)
        if f < 8:
           w= 10000
        else:
           w=1000
        sum_wights+=w
        sat_db[cluse_num]= [w, i,0 ]
        cluse_num += 1
    for i in range(var_num,not_var_num+1):
        #w=abs(int(np.random.normal(5000,100)))
        f = random.randint(0, 10)
        if f < 3:
           w= 10000
        else:
           w=1000
        sum_wights+=w
        sat_db[cluse_num]= [w, -i,0 ]
        cluse_num += 1
    #print(sat_db)
    tuple_lenght=len(D_features)+len(Y_features)+len(X_features)+1
    org_joined_df=org_joined_df.drop_duplicates()
    for i in  range(1, len(org_joined_df.index)):
        print('cluse numer :', i)
        #print(len(org_joined_df.index))

        row = org_joined_df.iloc[i].values
        #row=[7 ,'>12' ,0, 'Male', '>=70' ,18 ,'>12' ,1]

        #print(row)
        tmp=list2str(row[len(D_features)+1:len(D_features)+1+len(X_features)])
        first_cols = list2str(row[0:tuple_lenght])
        sec_cols = list2str(row[tuple_lenght:tuple_lenght+len(D_features)+1])+tmp+list2str(row[-len(Y_features):])
        #print(org_joined_df.columns)
        thris_col=list2str(row[0:len(D_features)+1])+tmp+list2str(row[-len(Y_features):])

        #print(first_cols)
        #print(sec_cols)
        #print(thris_col)
        in_key1=prov_in_by_value.get(first_cols)
        if not in_key1:
            in_key1 = prov_not_in_by_value.get(first_cols)
        in_key2=prov_in_by_value.get(sec_cols)
        if not in_key2:
            in_key2 = prov_not_in_by_value.get(sec_cols)
        why_not=prov_in_by_value.get(thris_col)
        if not why_not:
            why_not = prov_not_in_by_value.get(thris_col)
   #         prov_not_in_by_value.set(sec_cols, prov_conter)
    #        prov_not_in_by_idnt.set(prov_conter, sec_cols)
     #       why_not=prov_conter
        if why_not!=in_key1 and why_not!=in_key2:
            f=random.randint(0,10)
            if f<9:
             #pass
             sat_db[cluse_num]= [sum_wights+1, in_key1,in_key2,why_not, 0]
            else:
             #pass
             sat_db[cluse_num]= [1000, in_key1, in_key2, why_not, 0]
            cluse_num+=1

    num_var = n_in_db+not_var_num
    num_cls = len(sat_db.keys())

    file = open("sat.cnf", "w")
    file.write('p' + ' wcnf ' + str(num_var) + ' ' + str(num_cls) + ' '+str(sum_wights+1) + '\n')

    for key,value in sat_db.items():
        value=str(value)
        value=value.replace('[','')
        value=value.replace(']', '')
        value=value.split(',')

        #print(value[0])
        #v1=int(value[0])
        #print(len(value))
        if len(value)>3:
         line=str(value[0])+' -'+str(value[1]).replace(' ','')+' -'+str(value[2]).replace(' ','')+' '+str(value[3]).replace(' ','')+' 0 \n'
        else:
         line = str(value[0]) +' ' + str(value[1]).replace(' ', '') +' 0 \n'
        file.write(line)

from sklearn.utils import shuffle

def partition_grounder(data, path,sample_size, X, Y, Z=[], par_n=10):
    np.random.seed(32)  # for reproducible results.
    partition=np.array_split(data.reindex(np.random.permutation(data.index)), par_n)
    i=0
    for df in partition:
        repos, features = grounder4(df, sat, X, Y, Z)
        os.system('./open-wbo-master/./open-wbo ' + sat + ' >>' + out)
        df=data_from_sat(features, repos, out)
        if i>0:
           df0 = df0.append(df)
        else:
           df0 = df
    return df0


flatten = lambda *n: (e for a in n
    for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))

def advanced_grounder(data, indeps,smother=1,sample_size=100000,path='./sat.cnf'):
    #data=data[X+Y+Z]
    #data=data[X+Y+Z]
    prov_not_in_by_value=dict()
    prov_not_in_by_idnt=dict()
    prov_in_by_value=dict()
    prov_in_by_idnt=dict()
    sat_db=dict()
    repos=dict()
    print(data.columns)
    print(len(data.index))
    n_conflicts=0

    i=0
    for indep in indeps:  # this look converts bags to set wrt. all indepedances
        Z1 = indep[2]
        if Z1 == []:
            for indep in indeps:  # this look converts bags to set wrt. all indepedances
                    indep[2] = indep[2] + ['z' + str(i)]
                    data['z' + str(i)]='dummy'
        i+=1
    #print(data[Z[0]])
    indep_ounter=0

    features = list(set(flatten(indeps)))

    for indep in indeps:  # this look converts bags to set wrt. all indepedances
        X1 = indep[0]
        Y1 = indep[1]
        Z1 = indep[2]
        if indep_ounter==0:
            grouped=pd.DataFrame({'aaaa' : data.groupby(list(data.columns)).size()}).reset_index()
            grouped.reindex_axis(sorted(grouped.columns), axis=1)
            #print(grouped[1:10])
            #print(grouped.columns)
            columns=list(grouped.columns)
            columns.remove('aaaa')
            df = pd.DataFrame(columns= columns+['zzzz'+str(indep_ounter)])
            #df=df.reindex_axis(sorted(df.columns), axis=1)
        else:
            #features=['ZZZZZZZZZZ' + str(i) for i in range(0, indep_ounter)]+features
            grouped=grouped.iloc[0:0]
            #print(df[1:10])
            #print(df.columns)
            grouped=pd.DataFrame({'aaaa' : df.groupby(list(df.columns)).size()}).reset_index()
            grouped.reindex_axis(sorted(list(grouped.columns)), axis=1)
            #grouped.reindex_axis(sorted(grouped.columns), axis=1)
            #print(grouped[1:10])
            print(grouped.columns)

            #print(grouped)
            columns=list(grouped.columns)
            columns.remove('aaaa')
            df = pd.DataFrame(columns= columns+['zzzz'+str(indep_ounter)])

        indep_ounter+=1
        counter=1
        #grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        for i in range(0,len(grouped.index)):
             row=grouped.iloc[i].values
             k = 0
             #print(grouped.columns)
             for j in range(0, int(row[-1]/smother)): ##last column contains the frequencies
                 value=row[0:-1].tolist()
                 value.append(k)
                 df.loc[-1]=value
                 #print(df)
                 tmp = list2str(value)
                 #prov_in_by_value[tmp]=counter
                 #prov_in_by_idnt[counter]= tmp
                 #repos[counter]=tmp
                 #print(counter)
                 df.index = df.index + 1  # shifting index
                 k+=1
                 counter+=1
                 #df = df.sort_index()
    #print(df)
    df = df.reindex_axis(sorted(df.columns), axis=1)
    print('Bag Converted to set')
    var_num=counter
    #print(df.columns)
    #printDB(prov_in_by_value)
    #print(df)
    # K has been inserted to convert a bag to set
    counter=1

    for i in range(0, len(df.index)):
                 rowd = df.iloc[i]
                 row = df.iloc[i].values
                 tmp = list2str(row)
                 prov_in_by_value[tmp]=counter
                 prov_in_by_idnt[counter]= tmp
                 counter += 1
    print(rowd)
    prov_conter = len(prov_in_by_value.keys())
    for indep in indeps:  # this look converts bags to set wrt. all indepedances
        X1 = indep[0]
        Y1 = indep[1]
        Z1 = indep[2]
        org_joined_df = pd.merge(df, df, left_on=Z1, right_on=Z1, how='inner')
        print(org_joined_df.columns)
        joined_df=org_joined_df[sorted([item+'_x' for item in X1]+Z1+[item+'_y' for item in Y1]+['zzzz'+str(i)+'_x' for i in range(0,indep_ounter)])]
        print('joined_df',len(joined_df.index),joined_df.columns)
        joined_df.reset_index()
        joined_df.drop_duplicates()
        ###
        ## next part should be integrated with the conversion to sat
        ##
        print("####################################################################")
        for i in range(0,len(joined_df.index)):
             row=joined_df.iloc[i].values
             value=list2str(row)
             tmp=prov_in_by_value.get(value)
             if not tmp:
                 tmp = prov_not_in_by_value.get(value)
                 if not tmp:
                         prov_conter = prov_conter + 1
                         prov_not_in_by_value[value]= prov_conter
                         prov_not_in_by_idnt[prov_conter]= value
                         #repos[prov_conter]=value
        print(joined_df.iloc[i])
        not_var_num=prov_conter
        #printDB(prov_not_in_by_value)
        print(len(org_joined_df.index))
        #org_joined_df = pd.merge(joined_df, joined_df, left_on=Z, right_on=Z, how='inner') #3370284
        print('joined_df',len(org_joined_df.index))
        cluse_num=1
        sum_wights=0
        n_in_db=len(prov_in_by_idnt.keys())
        n_not_in_db=len(prov_not_in_by_idnt.keys())


        p#rint('Self joined performed')

        #print(sat_db)
        tuple_lenght=len(X1)+len(Y1)+len(Z1)+1
        org_joined_df.reset_index()
        org_joined_df=org_joined_df.drop_duplicates()
        print(len(org_joined_df.index))
        #print('after sample',len(org_joined_df.index))
        sum_wights=1000000000
        #print('number of conflicts',len(org_joined_df.index))
        #print('number of in', n_in_db)
        #print('number of out', n_not_in_db)
        if len(org_joined_df.index)>sample_size:
            org_joined_df=org_joined_df.sample(n=sample_size,replace=False)
        #print('number of cluses to explore',len(org_joined_df.index))
        for index, row in  org_joined_df.iterrows():
            #print(i)
            #print(len(org_joined_df.index))
            #print(index,row)
            lbll1=sorted([x + '_x' for x in X1]+['zzzz'+str(i)+'_x' for i in range(0,indep_ounter)] + [x + '_x' for x in Y1] + Z1)
            first_cols = list2str([row.get(lbl) for lbl in lbll1])
            lbll2=sorted([x + '_y' for x in X1] + ['zzzz'+str(i)+'_y' for i in range(0,indep_ounter)]+[x + '_y' for x in Y1] + Z1)
            sec_cols = list2str([row.get(lbl) for lbl in  lbll2])
            #print(org_joined_df.columns)
            lbll3 = sorted([x + '_x' for x in X1] + ['zzzz'+str(i)+'_x' for i in range(0,indep_ounter)]+[x + '_y' for x in Y1] + Z1)
            thris_col=list2str([row.get(lbl) for lbl in  lbll3])
            if first_cols==sec_cols or first_cols==thris_col or thris_col==sec_cols:
                continue
            #print(first_cols)
            #print(sec_cols)
            #print(thris_col)
            conf_flag=1
            in_key1=prov_in_by_value.get(first_cols)
            if in_key1 is None:
                in_key1 = prov_not_in_by_value.get(first_cols)
                conf_flag=0
            in_key2=prov_in_by_value.get(sec_cols)
            if in_key2 is None:
                in_key2 = prov_not_in_by_value.get(sec_cols)
                conf_flag = 0
            why_not=prov_not_in_by_value.get(thris_col)
            if  why_not is None:
                why_not = prov_in_by_value.get(thris_col)
                conf_flag = 0
       #         prov_not_in_by_value.set(sec_cols, prov_conter)
        #        prov_not_in_by_idnt.set(prov_conter, sec_cols)
         #       why_not=prov_conter
            #print(cluse_num)
            if why_not!=in_key1 and why_not!=in_key2:

                if in_key1 is not None and  in_key2  is not None and why_not is not None:
                    if conf_flag==1:
                        sat_db[cluse_num]= [int(sum_wights/(1000)), -in_key1, -in_key2, why_not, 0]
                        n_conflicts+=1
                    else:
                        sat_db[cluse_num] = [int(sum_wights / (1000)), -in_key1, -in_key2, why_not, 0]
                    cluse_num+=1
                else:
                    print('Something is wrong', in_key1, in_key2, why_not)
        print('proceeed', indep,len(sat_db.keys()))
    #ratio1=cluse_num/(n_in_db)
    #ratio2=  cluse_num/n_not_in_db
    #ratio=n_conflicts/(n_in_db+n_not_in_db)
    print('number of real conflict',n_conflicts)
    #ratio=1
    #if n_conflicts==0:
    #    n_conflicts=cluse_num

    ratio = n_conflicts / (n_in_db + n_not_in_db)
    if ratio==0:
        ratio=1
    for i in range(1, var_num):
        # w=abs(int(np.random.normal(1000,100)))
        f = random.randint(0, 10)
        w = int(int(sum_wights/10000000)/(ratio))
        #sum_wights += w
        sat_db[cluse_num] = [w, i, 0]
        cluse_num += 1
        #sat_db[cluse_num] = [1, -i, 0]
        #cluse_num += 1
        #sat_db[cluse_num] = [sum_wights, i, -i, 0]
        #cluse_num += 1
    for i in range(var_num, not_var_num + 1):
        # w=abs(int(np.random.normal(5000,100)))
        f = random.randint(0, 10)
        w = int(int(sum_wights/(10000000))/(ratio))
        #sum_wights += w
        #sat_db[cluse_num] = [1, i, 0]
        #print(sat_db[cluse_num])
        #cluse_num += 1
        sat_db[cluse_num] = [w, -i, 0]
        #print(sat_db[cluse_num])
        cluse_num += 1
        #sat_db[cluse_num] = [sum_wights, i,-i, 0]
        #print(sat_db[cluse_num])
        #cluse_num += 1

    num_var = n_in_db+not_var_num
    num_cls = len(sat_db.keys())

    file = open(path, "w")
    file.write('p' + ' wcnf ' + str(num_var) + ' ' + str(num_cls) + ' '+str(sum_wights) + '\n')

    for key,value in sat_db.items():
        value=str(value)
        value=value.replace('[','')
        value=value.replace(']', '')
        value=value.split(',')

        #print(value[0])
        #v1=int(value[0])
        #print(len(value))
        value=str(value)
        value=value.replace('[','')
        value=value.replace(']', '')
        value = value.replace("'", '')
        value = value.replace(',', '')
        line=str(value).replace('[','')+' \n'
        file.write(line)
    #print('Convertion to SAT is completed!')
    features= list(df.columns)
    features.reverse()
    return prov_not_in_by_idnt, prov_in_by_idnt,  df.columns


def calculate_area(row):
    return print(row)

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

def old_sing_grounder(data, X, Y, Z=[],path='./', sample_size=1, insert_ratio=1,conf_weight=1):

    prov_not_in_by_value=dict()      # a dictionary that keeps track of the tuples not in data by their value
    prov_not_in_by_idnt=dict()       # a dictionary that keeps track of the tuples not in data by their identifier
    prov_in_by_value=dict()          # a dictionary that keeps track of the tuples  in data by their value
    prov_in_by_idnt=dict()
    prov_in_by_www=dict()# a dictionary that keeps track of the tuples  in data by their identifier
    sat_db=dict()       # keeps list of cluses
    repos=dict()


    # convert bag to set by injecting k
    # create provenance of tuples in D
    grouped=pd.DataFrame({'count' : data.groupby( X+Z+Y ).size()}).reset_index()
    df = pd.DataFrame(columns=['k'] + X + Z + Y)
    #print(len(data.index))
    counter=1
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         #cp=CP(grouped, Y, [row[-1]], X+Z, row[0:-1])
         for j in range(0, int(row[-1])): ##last column contains the frequencies
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value[tmp]=counter
             prov_in_by_idnt[counter]= tmp
             prov_in_by_www[counter]=k
             repos[counter]=tmp
             df.index = df.index + 1  # shifting index
             k+=1
             counter+=1
    var_num=counter
    #print(len(df.index))


    df.reset_index()
    org_joined_df = pd.merge(df, df, left_on=Z, right_on=Z, how='inner')  #performs a self join on Z
    joined_df=org_joined_df[['k_x']+ [item+'_x' for item in X]+Z+[item+'_y' for item in Y]].drop_duplicates()
    joined_df.reset_index()     #self-joined data



    #create a dictionary of the provenance of tuples no in data but may
    prov_conter=len(prov_in_by_value.keys())+1
    for i in range(0,len(joined_df.index)):
         row=joined_df.iloc[i].values
         value=list2str(row)
         tmp=prov_in_by_value.get(value)
         if not tmp:
             prov_conter = prov_conter + 1
             prov_not_in_by_value[value]= prov_conter
             prov_not_in_by_idnt[prov_conter]= value
             repos[prov_conter]=value

    not_var_num=prov_conter
    org_joined_df = pd.merge(joined_df, joined_df, left_on=Z, right_on=Z, how='inner') #self join
    cluse_num=1
    n_in_db=len(prov_in_by_idnt.keys())
    n_not_in_db=len(prov_not_in_by_idnt.keys())
    tuple_lenght=len(X)+len(Y)+len(Z)+1
    org_joined_df=org_joined_df.drop_duplicates()
    sum_wights=100000   # sum of all the wights (used by maxsat solver)
    n_conflicts=0
    print('number of cluses to explore',len(org_joined_df.index))
    print(org_joined_df[['X_x_x','Y_y_x','Z']].drop_duplicates())
    # generate the conflicts
    naive_rep=[]
    rep2=-1
    rep1=-1
    for i in  range(1, len(org_joined_df.index)):
        row = org_joined_df.iloc[i].values
        freq=row[0]
        print(org_joined_df.columns)
        print(row)
        tmp=list2str(row[len(X)+1:len(X)+1+len(Z)])
        first_cols = list2str(row[0:tuple_lenght])
        sec_cols = list2str(row[tuple_lenght:tuple_lenght+len(X)+1])+tmp+list2str(row[-len(Y):])
        thris_col=list2str(row[0:len(X)+1])+tmp+list2str(row[-len(Y):])
        in_key1=prov_in_by_value.get(first_cols)
        if not in_key1:
            in_key1 = prov_not_in_by_value.get(first_cols)
        else:
            rep1=in_key1
        in_key2=prov_in_by_value.get(sec_cols)
        if not in_key2:
            in_key2 = prov_not_in_by_value.get(sec_cols)
        else:
            rep2=in_key2
        why_not=prov_not_in_by_value.get(thris_col)
        if not why_not:
            why_not = prov_in_by_value.get(thris_col)
        if why_not!=in_key1 and why_not!=in_key2:
                if rep1 not in naive_rep and rep2 not in naive_rep:
                  naive_rep.insert(0,rep1)
                #int(sum_wights / conf_weight)
                print(first_cols, sec_cols, thris_col)
                sat_db[cluse_num]= [conf_weight, -in_key1, -in_key2, why_not, 0]
                cluse_num=cluse_num+1
                n_conflicts=n_conflicts+1

    if sample_size<1:
        f=len(sat_db)
        n=sample_size * f

        from random import choices

        k = int(n)
        selected_keys = choices(list(sat_db.keys()), k=k)
        selected_sat_db=dict()
        for key in selected_keys:
            selected_sat_db[key]=sat_db[key]

        sat_db=selected_sat_db
    #sat_db=[]
    if n_in_db + n_not_in_db==0:
        ratio=1
    else:
        ratio = n_conflicts / (n_in_db + n_not_in_db)
    if ratio==0:
      ratio=1
    #print(len(prov_in_by_www.keys()))
    for i in range(1, var_num):
        #print(prov_in_by_www[i],w)
        w=int(float(sum_wights)/(1000000+int(prov_in_by_www[i])+1))
        sat_db[cluse_num] = [1, i, 0] #int(prov_in_by_www[i]/10)+
        cluse_num += 1
    for i in range(var_num+1, not_var_num + 1):
        #w = int(int(sum_wights))
        sat_db[cluse_num] = [sum_wights, -i, 0]
        cluse_num += 1


    #create a cnf file for maxsat solver
    num_var = n_in_db+not_var_num
    num_cls = len(sat_db.keys())
    pid = os.getpid()
    file = open(path+str(pid)+'.cnf', "w")
    #print(path+str(pid)+'.cnf', "w")
    file.write('p' + ' wcnf ' + str(num_var) + ' ' + str(num_cls) + ' '+str(sum_wights) + '\n')
    for key,value in sat_db.items():
        value=str(value)
        value=value.replace('[','')
        value=value.replace(']', '')
        value=value.split(',')
        value=str(value)
        value=value.replace('[','')
        value=value.replace(']', '')
        value = value.replace("'", '')
        value = value.replace(',', '')
        line=str(value).replace('[','')+' \n'
        file.write(line)
    print('Convertion to SAT is completed!: '+path+str(pid)+'.cnf')
    return prov_not_in_by_idnt, prov_in_by_idnt, naive_rep,path+str(pid)+'.cnf',df.columns

def grounder4(data, X, Y, Z=[],smother=1,path="../sat.cnf",sample_size=1, insert_ratio=1,in_weight=10000000,out_weight=10000000,conf_weight=10000000):
    #data=data[X+Y+Z]
    #data=data[X+Y+Z]
    prov_not_in_by_value=dict()
    prov_not_in_by_idnt=dict()
    prov_in_by_value=dict()
    prov_in_by_idnt=dict()
    sat_db=dict()
    repos=dict()
    #print(data.columns)
    #print(len(data.index))
    if  Z==[]:
        Z=['Z']
        data['Z']='aaaaa'
    #print(data[Z[0]])
    grouped=pd.DataFrame({'count' : data.groupby( X+Z+Y ).size()}).reset_index()
    #print(grouped)
    df = pd.DataFrame(columns=['k'] + X + Z + Y)

    #print('Dics cleared')
    #grouped=grouped.reset_index()
    #ate = data.groupby(['origin', 'carrier'])['delayed'].mean()
    #ate.columns = ['Z', 'X', 'Y']
    counter=1
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         #print(grouped.columns)
         for j in range(0, int(row[-1]/smother)): ##last column contains the frequencies
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value[tmp]=counter
             prov_in_by_idnt[counter]= tmp
             repos[counter]=tmp
             #print(counter)
             df.index = df.index + 1  # shifting index
             k+=1
             counter+=1
             #df = df.sort_index()
    #print('Prov in dic created')
    var_num=counter
    #printDB(prov_in_by_value)
    #print(df)
    # K has been inserted to convert a bag to set
    df.reset_index()
    org_joined_df = pd.merge(df, df, left_on=Z, right_on=Z, how='inner')
    joined_df=org_joined_df[['k_x']+ [item+'_x' for item in X]+Z+[item+'_y' for item in Y]].drop_duplicates()
    #print('joined_df',len(joined_df.index))
    joined_df.reset_index()

    '''
    counter=0
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         for j in range(0, row[-1]):
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value.set(tmp, counter)
             prov_in_by_idnt.set(counter, tmp)
             counter+=1
    '''

    prov_conter=len(prov_in_by_value.keys())
    for i in range(0,len(joined_df.index)):
         row=joined_df.iloc[i].values
         value=list2str(row)
         tmp=prov_in_by_value.get(value)
         if not tmp:
             prov_conter = prov_conter + 1
             prov_not_in_by_value[value]= prov_conter
             prov_not_in_by_idnt[prov_conter]= value
             repos[prov_conter]=value

    not_var_num=prov_conter
    #printDB(prov_not_in_by_value)
    #print(len(org_joined_df.index))
    org_joined_df = pd.merge(joined_df, joined_df, left_on=Z, right_on=Z, how='inner') #3370284
    print('joined_df',len(org_joined_df.index))
    cluse_num=1
    sum_wights=0
    n_in_db=len(prov_in_by_idnt.keys())
    n_not_in_db=len(prov_not_in_by_idnt.keys())


    #print('Self joined performed')

    #print(sat_db)
    tuple_lenght=len(X)+len(Y)+len(Z)+1
    org_joined_df=org_joined_df.drop_duplicates()
    #print(len(org_joined_df.index))
    #print('after sample',len(org_joined_df.index))
    sum_wights=1000000000
    #print('number of conflicts',len(org_joined_df.index))
    #print('number of in', n_in_db)
    #print('number of out', n_not_in_db)
    #org_joined_df=org_joined_df.sample(frac=sample_size,replace=False)

    n_conflicts=0
    print('number of cluses to explore',len(org_joined_df.index))
    for i in  range(1, len(org_joined_df.index)):
        #print(i)
        #print(len(org_joined_df.index))

        row = org_joined_df.iloc[i].values
        #row=[7 ,'>12' ,0, 'Male', '>=70' ,18 ,'>12' ,1]

        #print(row)
        tmp=list2str(row[len(X)+1:len(X)+1+len(Z)])
        first_cols = list2str(row[0:tuple_lenght])
        sec_cols = list2str(row[tuple_lenght:tuple_lenght+len(X)+1])+tmp+list2str(row[-len(Y):])
        #print(org_joined_df.columns)
        thris_col=list2str(row[0:len(X)+1])+tmp+list2str(row[-len(Y):])

        #print(first_cols)
        #print(sec_cols)
        #print(thris_col)
        conf_flag=1
        in_key1=prov_in_by_value.get(first_cols)
        if not in_key1:
            in_key1 = prov_not_in_by_value.get(first_cols)
            conf_flag=0
        in_key2=prov_in_by_value.get(sec_cols)
        if not in_key2:
            in_key2 = prov_not_in_by_value.get(sec_cols)
            conf_flag = 0
        why_not=prov_not_in_by_value.get(thris_col)
        if not why_not:
            why_not = prov_in_by_value.get(thris_col)
            conf_flag = 0
   #         prov_not_in_by_value.set(sec_cols, prov_conter)
    #        prov_not_in_by_idnt.set(prov_conter, sec_cols)
     #       why_not=prov_conter
        if why_not!=in_key1 and why_not!=in_key2:
            f=random.randint(0,10)

            if conf_flag==1:
                sat_db[cluse_num]= [int(sum_wights/(conf_weight)), -in_key1, -in_key2, why_not, 0]
                n_conflicts+=1
            else:
                sat_db[cluse_num] = [int(sum_wights / (conf_weight)), -in_key1, -in_key2, why_not, 0]

            cluse_num+=1

    #ratio1=cluse_num/(n_in_db)
    #ratio2=  cluse_num/n_not_in_db
    #ratio=n_conflicts/(n_in_db+n_not_in_db)
    ratio = n_conflicts / (n_in_db + n_not_in_db)
    if ratio==0:
        ratio=1
    #print('number of real conflict',n_conflicts)
    for i in range(1, var_num):
        # w=abs(int(np.random.normal(1000,100)))
        f = random.randint(0, 10)
        w = int(int(sum_wights/in_weight)/(ratio))
        #sum_wights += w
        sat_db[cluse_num] = [w, i, 0]
        cluse_num += 1
        #sat_db[cluse_num] = [1, -i, 0]
        #cluse_num += 1
        #sat_db[cluse_num] = [sum_wights, i, -i, 0]
        #cluse_num += 1
    for i in range(var_num, not_var_num + 1):
        # w=abs(int(np.random.normal(5000,100)))
        f = random.randint(0, 10)
        w = int(int(sum_wights/(in_weight))*insert_ratio)
        #sum_wights += w
        #sat_db[cluse_num] = [1, i, 0]
        #print(sat_db[cluse_num])
        #cluse_num += 1
        sat_db[cluse_num] = [w, -i, 0]
        #print(sat_db[cluse_num])
        cluse_num += 1
        #sat_db[cluse_num] = [sum_wights, i,-i, 0]
        #print(sat_db[cluse_num])
        #cluse_num += 1

    num_var = n_in_db+not_var_num
    num_cls = len(sat_db.keys())

    file = open(path, "w")
    file.write('p' + ' wcnf ' + str(num_var) + ' ' + str(num_cls) + ' '+str(sum_wights) + '\n')

    for key,value in sat_db.items():
        value=str(value)
        value=value.replace('[','')
        value=value.replace(']', '')
        value=value.split(',')

        #print(value[0])
        #v1=int(value[0])
        #print(len(value))
        value=str(value)
        value=value.replace('[','')
        value=value.replace(']', '')
        value = value.replace("'", '')
        value = value.replace(',', '')
        line=str(value).replace('[','')+' \n'
        file.write(line)
    #print('Convertion to SAT is completed!')
    return prov_not_in_by_idnt, prov_in_by_idnt, [], path,df.columns


def sing_grounder(data, X, Y, Z=[],path='./', sample_size=1, insert_ratio=1,conf_weight=1):

    prov_not_in_by_value=dict()      # a dictionary that keeps track of the tuples not in data by their value
    prov_not_in_by_idnt=dict()       # a dictionary that keeps track of the tuples not in data by their identifier
    prov_in_by_value=dict()          # a dictionary that keeps track of the tuples  in data by their value
    prov_in_by_idnt=dict()
    prov_in_by_www=dict()# a dictionary that keeps track of the tuples  in data by their identifier
    sat_db=dict()       # keeps list of cluses
    repos=dict()


    # convert bag to set by injecting k
    # create provenance of tuples in D
    grouped=pd.DataFrame({'count' : data.groupby( X+Z+Y ).size()}).reset_index()
    df = pd.DataFrame(columns=['k'] + X + Z + Y)
    #print(len(data.index))
    counter=1
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         #cp=CP(grouped, Y, [row[-1]], X+Z, row[0:-1])
         for j in range(0, int(row[-1])): ##last column contains the frequencies
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value[tmp]=counter
             prov_in_by_idnt[counter]= tmp
             prov_in_by_www[counter]=k
             repos[counter]=tmp
             df.index = df.index + 1  # shifting index
             k+=1
             counter+=1
    var_num=counter
    #print(len(df.index))


    df.reset_index()
    org_joined_df = pd.merge(df, df, left_on=Z, right_on=Z, how='inner')  #performs a self join on Z
    joined_df=org_joined_df[['k_x']+ [item+'_x' for item in X]+Z+[item+'_y' for item in Y]].drop_duplicates()
    joined_df.reset_index()     #self-joined data



    #create a dictionary of the provenance of tuples no in data but may
    prov_conter=len(prov_in_by_value.keys())+1
    for i in range(0,len(joined_df.index)):
         row=joined_df.iloc[i].values
         value=list2str(row)
         tmp=prov_in_by_value.get(value)
         if not tmp:
             prov_conter = prov_conter + 1
             prov_not_in_by_value[value]= prov_conter
             prov_not_in_by_idnt[prov_conter]= value
             repos[prov_conter]=value

    not_var_num=prov_conter
    #org_joined_df = pd.merge(joined_df, joined_df, left_on=Z, right_on=Z, how='inner') #self join
    cluse_num=1
    n_in_db=len(prov_in_by_idnt.keys())
    n_not_in_db=len(prov_not_in_by_idnt.keys())
    tuple_lenght=len(X)+len(Y)+len(Z)+1
    #org_joined_df=org_joined_df.drop_duplicates()
    sum_wights=1000000000   # sum of all the wights (used by maxsat solver)
    n_conflicts=0
    print('number of clauses to explore',len(org_joined_df.index))
    #print(org_joined_df[['X_x_x','Y_y_x','Z']].drop_duplicates())
    # generate the conflicts
    naive_rep=[]
    rep2=-1
    rep1=-1
    for i in  range(1, len(org_joined_df.index)):
        row = org_joined_df.iloc[i].values
        freq=row[0]
        #print(org_joined_df.columns)
        #print(row)
        tmp=list2str(row[len(X)+1:len(X)+1+len(Z)])
        first_cols = list2str(row[0:tuple_lenght])
        sec_cols = list2str(row[tuple_lenght:tuple_lenght+len(X)+1])+tmp+list2str(row[-len(Y):])
        thris_col=list2str(row[0:len(X)+1])+tmp+list2str(row[-len(Y):])
        in_key1=prov_in_by_value.get(first_cols)
        if not in_key1:
            in_key1 = prov_not_in_by_value.get(first_cols)
        else:
            rep1=in_key1
        in_key2=prov_in_by_value.get(sec_cols)
        if not in_key2:
            in_key2 = prov_not_in_by_value.get(sec_cols)
        else:
            rep2=in_key2
        why_not=prov_not_in_by_value.get(thris_col)
        if not why_not:
            why_not = prov_in_by_value.get(thris_col)
        if why_not!=in_key1 and why_not!=in_key2:
                if rep1 not in naive_rep and rep2 not in naive_rep:
                  naive_rep.insert(0,rep1)
                #int(sum_wights / conf_weight)
                if sample_size < 2:
                    f = random.randint(0, 10)
                    if f < 8:
                        w = 5000
                    else:
                        w = 50000
                else:
                      w=sum_wights
                sat_db[cluse_num]= [w, -in_key1, -in_key2, why_not, 0]  # COMPAS generated with 1000
                cluse_num=cluse_num+1
                n_conflicts=n_conflicts+1

    if sample_size<1:
        f=len(sat_db)
        n=sample_size * f

        from random import choices

        k = int(n)
        selected_keys = choices(list(sat_db.keys()), k=k)
        selected_sat_db=dict()
        for key in selected_keys:
            selected_sat_db[key]=sat_db[key]

        sat_db=selected_sat_db
    #sat_db=[]
    if n_in_db + n_not_in_db==0:
        ratio=1
    else:
        ratio = n_conflicts / (n_in_db + n_not_in_db)
    if ratio==0:
      ratio=1
    #print(len(prov_in_by_www.keys()))
    for i in range(1, var_num):
        #print(prov_in_by_www[i],w)
        w=int(float(sum_wights)/(1000000+int(prov_in_by_www[i])+1))
        f = random.randint(0, 10)
        if f < 8:
            w = 60000
        else:
            w = 100000

        sat_db[cluse_num] = [w, i, 0] #int(prov_in_by_www[i]/10)+
        cluse_num += 1
    for i in range(var_num+1, not_var_num + 1):
        #w = int(int(sum_wights))
        f = random.randint(0, 10)
        if f < 8:
            w = 10000
        else:
            w = 1000

        sat_db[cluse_num] = [500000, -i, 0]
        cluse_num += 1


    #create a cnf file for maxsat solver
    num_var = n_in_db+not_var_num
    num_cls = len(sat_db.keys())
    pid = os.getpid()
    file = open(path+str(pid)+'.cnf', "w")
    #print(path+str(pid)+'.cnf', "w")
    file.write('p' + ' wcnf ' + str(num_var) + ' ' + str(num_cls) + ' '+str(sum_wights) + '\n')
    for key,value in sat_db.items():
        value=str(value)
        value=value.replace('[','')
        value=value.replace(']', '')
        value=value.split(',')
        value=str(value)
        value=value.replace('[','')
        value=value.replace(']', '')
        value = value.replace("'", '')
        value = value.replace(',', '')
        line=str(value).replace('[','')+' \n'
        file.write(line)
    print('Convertion to SAT is completed!: '+path+str(pid)+'.cnf')
    return prov_not_in_by_idnt, prov_in_by_idnt, naive_rep,path+str(pid)+'.cnf',df.columns

def grounder2(data, storage, X, Y, Z=[]):
    #data=data[X+Y+Z]
    #data=data[X+Y+Z]

    prov_not_in_by_value=storage[0]
    prov_not_in_by_idnt=storage[1]
    prov_in_by_value=storage[2]
    prov_in_by_idnt=storage[3]
    sat_db=storage[4]
    flashDB(prov_not_in_by_value)
    flashDB(prov_not_in_by_idnt)
    flashDB(prov_in_by_value)
    flashDB(prov_in_by_idnt)
    flashDB(sat_db)
    grouped=pd.DataFrame({'count' : data.groupby( X+Y+Z ).size()}).reset_index()
    df = pd.DataFrame(columns=['k']+X + Y + Z)
    print('Dics cleared')
    #grouped=grouped.reset_index()
    #ate = data.groupby(['origin', 'carrier'])['delayed'].mean()
    #ate.columns = ['Z', 'X', 'Y']
    counter=1
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         for j in range(0, int(row[-1])):
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value.set(tmp, counter)
             prov_in_by_idnt.set(counter, tmp)
             df.index = df.index + 1  # shifting index
             k+=1
             counter+=1
             #df = df.sort_index()
    print('Prov in dic created')
    var_num=counter
    printDB(prov_in_by_value)
    print('joined_df')
    # K has been inserted to convert a bag to set
    df.reset_index()
    joined_df=pd.merge(df,df,left_on=Z, right_on=Z, how='outer')
    joined_df=joined_df[['k_x', X[0]+'_x', Y[0]+'_y']+Z].drop_duplicates()
    print('joined_df')
    joined_df.reset_index()

    '''
    counter=0
    for i in range(0,len(grouped.index)):
         row=grouped.iloc[i].values
         k = 0
         for j in range(0, row[-1]):
             value=np.insert(row[0:-1],0,k)
             df.loc[-1]=value
             tmp = list2str(value)
             prov_in_by_value.set(tmp, counter)
             prov_in_by_idnt.set(counter, tmp)
             counter+=1
    '''

    prov_conter=prov_in_by_value.dbsize()
    for i in range(0,len(joined_df.index)):
         row=joined_df.iloc[i].values
         value=list2str(row)
         tmp=prov_in_by_value.get(value)
         if not tmp:
             prov_conter = prov_conter + 1
             prov_not_in_by_value.set(value, prov_conter)
             prov_not_in_by_idnt.set(prov_conter, value)

    not_var_num=prov_conter
    #printDB(prov_not_in_by_value)

    joined_df = pd.merge(joined_df, joined_df, left_on=Z, right_on=Z, how='outer')
    cluse_num=1
    sum_wights=0
    n_in_db=prov_in_by_idnt.dbsize()
    n_not_in_db=prov_not_in_by_idnt.dbsize()
    ratio=n_not_in_db/n_in_db

    print('Self joined performed')
    for i in range(1,var_num):
        #w=abs(int(np.random.normal(1000,100)))
        f = random.randint(0, 10)
        if f < 8:
           w= 10000
        else:
           w=1000
        sum_wights+=w
        sat_db.set(cluse_num, [w, i,0 ])
        cluse_num += 1
    for i in range(var_num,not_var_num+1):
        #w=abs(int(np.random.normal(5000,100)))
        f = random.randint(0, 10)
        if f < 3:
           w= 10000
        else:
           w=1000
        sum_wights+=w
        sat_db.set(cluse_num, [w, -i,0 ])
        cluse_num += 1
    #print(sat_db)


    for i in  range(1, len(joined_df.index)):
        print(i)
        #print(len(joined_df.index))

        row = joined_df.iloc[i].values
        #row=[7 ,'>12' ,0, 'Male', '>=70' ,18 ,'>12' ,1]
        #print(row)
        tmp=list2str(row[3:-3])
        first_cols = list2str(row[0:3])+tmp
        sec_cols = list2str(row[-3:])+tmp
        thris_col=list2str(row[0:2])+list2str([row[-1]])+tmp
        in_key1=prov_in_by_value.get(first_cols)
        if not in_key1:
            in_key1 = prov_not_in_by_value.get(first_cols)
        in_key2=prov_in_by_value.get(sec_cols)
        if not in_key2:
            in_key2 = prov_not_in_by_value.get(sec_cols)
        why_not=prov_in_by_value.get(thris_col)
        if not why_not:
            why_not = prov_not_in_by_value.get(thris_col)
   #         prov_not_in_by_value.set(sec_cols, prov_conter)
    #        prov_not_in_by_idnt.set(prov_conter, sec_cols)
     #       why_not=prov_conter
        if why_not!=in_key1 and why_not!=in_key2:
            f=random.randint(0,10)
            if f<8:
             #pass
             sat_db.set(cluse_num, [sum_wights+1, int(in_key1),int(in_key2),int(why_not), 0])
            else:
             #pass
             sat_db.set(cluse_num, [1000, int(in_key1), int(in_key2), int(why_not), 0])
            cluse_num+=1









def grounder(data, prov_not_in_by_value, prov_not_in_by_idnt, prov_in_by_value, prov_in_by_idnt,sat_db, X, Y, Z=[]):
    #data=data[X+Y+Z]
    grouped=data.groupby(X+Y+Z).count()
    grouped=grouped.reset_index()
    #ate = data.groupby(['origin', 'carrier'])['delayed'].mean()
    #ate.columns = ['Z', 'X', 'Y']
    #for group, name in grouped:
    for item in grouped.columns:
        if item not in X and item not in Y and item not in Z:
            break
    count_index=item
    for i in range(1,len(grouped.index)+1):
        first_row_index=i
        first_tuple=grouped.iloc[first_row_index-1:first_row_index]
        count=list(first_tuple[count_index])[0]
        for first_id in range(1,count+1):
            #print(row_index,id)
            for j in range(1, len(grouped.index)+1):
                second_row_index=j
                if first_row_index!=second_row_index:
                    second_tuple = grouped.iloc[second_row_index - 1:second_row_index]
                    sec_count = list(second_tuple[count_index])[0]
                    if first_tuple[Z].values==second_tuple[Z].values:
                        for second_id in range(1, sec_count+1):


                            tuple1=(str(first_id)+'#'+ str((first_tuple[X].values[0])[0])+'#'+str((first_tuple[Y].values[0])[0])+'#'+str((first_tuple[Z].values[0])[0]))
                            tuple2 = (str(second_id) + '#' + str((second_tuple[X].values[0])[0]) + '#' + str(
                                (second_tuple[Y].values[0])[0]) + '#' + str((second_tuple[Z].values[0])[0]))

                            print('#######')
                            print('pairs',tuple1,tuple2)
                            print('#######')

                            x=str((first_tuple[X].values[0])[0])
                            y=str((second_tuple[Y].values[0])[0])
                            z=str((first_tuple[Z].values[0])[0])

                            #clause=(first_row_index,first_id) (first_row_index,first_id))
                            value1=(str(first_id)+'#'+ x+'#'+y+'#'+z)
                            #key1=2^first_id+3^second_id
                            print(value1)
                            x=str((second_tuple[X].values[0])[0])
                            y=str((first_tuple[Y].values[0])[0])
                            z=str((second_tuple[Z].values[0])[0])
                            value2=(str(second_id)+'#'+ x+'#'+y+'#'+z)
                            print(value1)

                            in_key1=(2 ** first_row_index)*(3 **first_id)
                            in_key2 = (2  ** second_row_index)*(3 ** second_id)
                            prov_in_by_idnt.set(in_key1, tuple1)
                            prov_in_by_value.set(tuple1, in_key1)
                            prov_in_by_idnt.set(in_key2, tuple2)
                            prov_in_by_value.set(tuple2, in_key1)
                            not_in_index=prov_in_by_value.get(value1)
                            if not not_in_index:
                               tmp_key=prov_not_in_by_value.dbsize() + 1

                               if not prov_not_in_by_value.get(value1):
                                   prov_not_in_by_value.set(value1, tmp_key)
                                   prov_not_in_by_idnt.set(tmp_key, value1)

                               prov_in_by_idnt.set(in_key1, tuple1)
                               prov_in_by_value.set(tuple1, in_key1)
                               prov_in_by_idnt.set(in_key2, tuple2)
                               prov_in_by_value.set(tuple2, in_key1)
                               tmp = int.from_bytes(prov_not_in_by_value.get(value1), byteorder='little')
                               sat_db.set(sat_db.dbsize() + 1, [in_key1,in_key2,tmp])


                            not_in_index = prov_in_by_value.get(value2)
                            if not not_in_index:
                               tmp_key=prov_not_in_by_value.dbsize() + 1
                               if not prov_not_in_by_value.get(value2):
                                   prov_not_in_by_value.set(value2, tmp_key)
                                   prov_not_in_by_idnt.set(tmp_key, value2)

                               prov_in_by_idnt.set(in_key1, tuple1)
                               prov_in_by_value.set(tuple1, in_key1)
                               prov_in_by_idnt.set(in_key2, tuple2)
                               prov_in_by_value.set(tuple2, in_key1)
                               tmp=int.from_bytes(prov_not_in_by_value.get(value2),byteorder='little')
                               sat_db.set(sat_db.dbsize() + 1, [in_key2,in_key1,tmp] )



# p variable cluase



                            #db.set(value1, str(value).encode())

def printDB(db):
        print('###########')
        for key in db.scan_iter():
            # delete the key
            #value=r1.delete(key)
            value = db.get(key)
            print('key',key)
            print('value',str(value))



def flashDB(db):
    db.flushdb()
    #for key in db.scan_iter():
    #    db.delete(key)

from sklearn.svm import SVC

if __name__ == '__main__':
    ## start redis-server

        data = pd.read_csv("/Users/babakmac/Documents/XDBData/binadult.csv")
        out = '/Users/babakmac/Documents/FairDB/Core/sol2.txt'
        sat = '/Users/babakmac/Documents/FairDB/Core/sat2.cnf'
        features = ['education', 'occupation', 'age', 'race', 'sex', 'income','maritalstatus','hoursperweek'] # ,'capitalloss','capitalgain','workclass' 'hoursperweek',
        data=data[features]
        #D1_features = ['sex']
        #Y1_features = ['income','race']
        #X1_features = ['education', 'hoursperweek','occupation', 'age', 'maritalstatus','capitalloss','capitalgain','workclass']
        D1_features = ['race', 'sex', 'maritalstatus']
        Y1_features = ['income', 'occupation', 'age']
        X1_features = ['hoursperweek', 'education']

        D2_features = ['age']
        Y2_features = ['income', 'race', 'sex', 'maritalstatus']
        X2_features = ['education', 'occupation', 'hoursperweek']

        D1 = [D1_features, Y1_features, X1_features]
        D2 = [D2_features, Y2_features, X2_features]
        indeps = [D1, D2]


        D1_features = ['race', 'sex', 'maritalstatus']
        Y1_features = ['income']
        X1_features = ['hoursperweek', 'education']

        D2_features = ['age','race', 'sex','maritalstatus']
        Y2_features = ['income',]
        X2_features = ['education','occupation', 'hoursperweek']


        print(len(data.index))
        data = pd.read_csv("/Users/babakmac/Documents/fainess/FairDB/Data/Adult/train_0.csv")
        path='/Users/babakmac/Documents/fainess/FairDB/Data/Adult/'
        data=data.sample(frac=0.01)
        #data[features].to_csv('/Users/babakmac/Documents/FairDB/Experiments/Data/adult/26June/org_special_multi_0.csv')
        prov_not_in_by_idnt, prov_in_by_idnt, cnf, features = sing_grounder(data, D2_features,Y2_features,X2_features,path)
        os.system('/Users/babakmac/Documents/FairDB/Core/SatSolvers/open-wbo-master/./open-wbo ' + cnf + ' >>' + cnf+'out')
        #df = data_from_sat(features, prov_not_in_by_idnt, prov_in_by_idnt, out, path, insert_ratio=1, chunk_size)
        #print(len(df.index))
        #df.to_csv('/Users/babakmac/Documents/FairDB/Experiments/Data/adult/special_multi_0.csv')
