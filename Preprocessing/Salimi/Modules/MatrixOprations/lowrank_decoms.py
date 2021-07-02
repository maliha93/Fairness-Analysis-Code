
#import cvxpy as cvx
import numpy as np
import math
import decimal
from sklearn.decomposition import NMF

def low_rank_approx(A=None, r=1,loss='frobenius'):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    if not A.size:
        return A
    SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    if r>np.size(s):
        r=np.size(s)-1
    for i in range(r,np.size(s)):
        s[i]=0
    Ar = np.matmul(np.matmul(u, np.diag(s)), v)
    #model=NMF(1,beta_loss=loss,init='nndsvd', solver='mu')

    #model=NMF(1,beta_loss=loss, solver='mu')

    #W=model.fit_transform(A)
    #H=model.components_
    #print('SVD: \n', Ar)
    #print('NMF: \n',W,H, W *H)


    return Ar


''' 
def smart_round(A,total):
    x = Int(len(A))
    objective = Minimize(sum_squares(A - x))
    constraints = [sum(x)==total,x>=0]
    prob = Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    #print('hellooo Bob',x.value)
    x=np.array(x.value)
    x=list(x)
    #print(sum(x))
    answer= [int(e[0]+0.01) for e in x]
    #for e in  x:
    #    print(int(e[0]+0.00001))
    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    #print(sum(x),sum(answer),x,answer)
    return answer
'''

if __name__ == "__main__":
    """
    Test: visualize an r-rank approximation of `lena`
    for increasing values of r
    Requires: scipy, matplotlib
    """
    w, h = 8, 5;
    x = [[0 for x in range(w)] for y in range(h)]
    #X= [[0,1],[1,0]]
    X = [[8,2],[2,2]]
    print(low_rank_approx(X))
    #X_a = np.dot(np.dot(P, np.diag(D)), Q)
    #print(X)
    m = 30
    n = 20
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    A=[0.9858930669021032, 0.0, 0.9858930669021032, 0.0, 0.0, 0.9858930669021032, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9858930669021032, 0.9858930669021032, 0.9858930669021032, 0.0, 0.0, 0.0, 0.9858930669021032, 0.9858930669021032, 0.9858930669021032, 0.0, 0.0, 0.0, 0.0, 0.9858930669021032, 0.9858930669021032, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9858930669021032, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9858930669021032, 0.0, 0.0, 0.9858930669021032, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9858930669021032, 0.0, 0.9858930669021032, 0.9858930669021032, 0.0, 0.9858930669021032, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9858930669021032, 0.0, 0.9858930669021032, 0.0, 0.0]
#    smart_round(A,19)

    np.random.seed(0)

    # Generate random data matrix A.
    m = 2
    n = 2
    k = 1
    A = np.random.rand(m, k).dot(np.random.rand(k, n))
    A=[[10,2],[20,50]]
    # Initialize Y randomly.
    Y_init = np.random.rand(m, k)
    # Construct the problem.

    np.random.seed(0)

    # Generate random data matrix A.

    #A = np.random.rand(m, k).dot(np.random.rand(k, n))

    # Initialize Y randomly.
    Y_init = np.random.rand(m, k)

    # Ensure same initial random Y, rather than generate new one
    # when executing this cell.
    Y = Y_init

    # Perform alternating minimization.
    MAX_ITERS = 30
    residual = np.zeros(MAX_ITERS)
    for iter_num in range(1, 1 + MAX_ITERS):
        # At the beginning of an iteration, X and Y are NumPy
        # array types, NOT CVXPY variables.

        # For odd iterations, treat Y constant, optimize over X.
        if iter_num % 2 == 1:
            X = cvx.Variable(k, n)
            constraint = [X >= 0]
        # For even iterations, treat X constant, optimize over Y.
        else:
            Y = cvx.Variable(m, k)
            constraint = [Y >= 0]

        # Solve the problem.
        obj = cvx.Minimize(cvx.norm(A - Y * X, 'fro'))
        prob = cvx.Problem(obj, constraint)
        prob.solve(solver=cvx.SCS)

        #if prob.status != cvx.OPTIMAL:
        #    raise Exception("Solver did not converge!")

        print( 'Iteration {}, residual norm {}'.format(iter_num, prob.value))
        residual[iter_num - 1] = prob.value

        # Convert variable to NumPy array constant for next iteration.
        if iter_num % 2 == 1:
            X = X.value
        else:
            Y = Y.value
    import matplotlib.pyplot as plt

    # Show plot inline in ipython.

    # Set plot properties.
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 16}
    plt.rc('font', **font)

    # Create the plot.
    plt.plot(residual)
    plt.xlabel('Iteration Number')
    plt.ylabel('Residual Norm')
    #plt.show()

    #
    # Print results.
    #
    print('Original matrix: \n',np.asmatrix(A))
    print('Left factor Y: \n',Y)
    print('Right factor X:\n', X)
    print('approximate: \n',Y * X)
    print('SVD \n',low_rank_approx(A,1))
    print('Residual A - Y * X:\n',A - Y * X)
    print('Residual A - LR', A - low_rank_approx(A,1))
    model=NMF(1)
    W=model.fit_transform(A)
    H=model.components_
    print('NMF \n',W,H, W *H)

    #print('Residual after {} iterations: {}'.format(iter_num, prob.value))

''' 
    M= [[1,4],[4,1]]
    Msr = [[25/10, 25/10], [25/10, 25/10]]
    X = Variable(2, 2)
    objective = Minimize(sum_squares(M-X))
    g1=norm(M,2).value
    g2 = norm(X, 2).value
    constraints = [norm(X-Msr, 'nuc')<0]
    prob = Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    #M=np.abs(M - X)

    print('hereeeeeeee',X.value,norm(X-Msr, 'nuc').value)

    


    
    i = 1
    #pylab.figure()
    #pylab.ion()
    while i < len(x) - 1:
        y = low_rank_approx((u, s, v), r=i)
        #pylab.imshow(y, cmap=pylab.cm.gray)
        #pylab.draw()
        i += 1
        #print percentage of singular spectrum used in approximation
        print("%0.2f %%" % (100 * i / 512.))
    '''