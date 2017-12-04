import csv
import numpy as np
import numpy.linalg as la
import timeit
import scipy
import numpy.random as rn

def Talpha(A, alpha, d1, d2):
    #d1 is the number of rows
    #This takes A to it's incomplete projection where each row/column has 
    #alpha % non-zero elements
    #The projection is partial in that it deletes more than it has to
    Aabs = abs(A)
    n1 = np.floor(alpha*d1).astype(int) #number of elements allowed in each column
    n2 = np.floor(alpha*d2).astype(int)
    row_nth_largest = np.partition(Aabs, -n2, axis = 1)[:, -n2] #nth largest element in each row
    col_nth_largest = np.partition(Aabs, -n1, axis = 0)[-n1]
    col_rep = np.dot(np.ones([d1,1]), col_nth_largest)
    row_rep = np.dot(row_nth_largest, np.ones([1,d2]))
    limiting = np.maximum(col_rep, row_rep)
    TA = np.multiply(np.multiply(Aabs,(Aabs >= limiting)),np.sign(A))
    return TA

def UVproj(A, bound, d):
    #Scales each row down to the bound
    for i in xrange(d):
        rownorm = la.norm(A[i])
        if rownorm > bound:
            A[i] = A[i]*bound/rownorm
    return A
            

def FRPCA(Y, alpha, gamma, eta, mu, r, T):
    d1 = np.size(Y[:,0])
    d2 = np.size(Y[0])
    S_init = Talpha(Y, alpha, d1, d2)
    M0 = Y - S_init
    [L, sigma, R] = la.svd(M0)
    L = L[:,0:r]
    sigmaroot = np.sqrt(sigma[0:r])
    opnorm = sigmaroot[0] #op norm is largest S.V.
    R = R[0:r]
    U0 = np.multiply(L,sigmaroot) 
    V0 = np.multiply(sigmaroot,np.transpose(R))
    Ubound = np.sqrt(2.*mu*r/d1)*opnorm #Used in projection
    Vbound = np.sqrt(2.*mu*r/d2)*opnorm
    Ut = U0
    Vt = V0
    gamalph = gamma*alpha
    Ynorm = la.norm(Y)#used for error
    for i in xrange(T):
        Mt = np.dot(Ut, np.transpose(Vt))    #Current est for low rank matrix
        St = Talpha(Y - Mt, gamalph, d1, d2) #Current est for sparse matrix
        lossmat = Mt+St-Y             #difference between est and observation
        error = la.norm(lossmat)/Ynorm
        print(error, 'error at iteration', i)
        UminV = np.dot(np.transpose(Ut), Ut) - np.dot(np.transpose(Vt), Vt)
        Utnew = Ut - eta*np.dot(lossmat, Vt)-.5*eta*np.dot(Ut, UminV)
        Vtnew = Vt - eta*np.dot(np.transpose(lossmat), Ut)-.5*eta*np.dot(Vt, -UminV)
        Ut = UVproj(Utnew, Ubound, d1)
        Vt = UVproj(Vtnew, Vbound, d2)
    return Ut, Vt

def main():
    rank = 5 #rank of Y, pixels in video
    d1 = 500 #rows
    d2 = 600 #columns
    alpha = .1 #corruption rate
    T = 50 #iterations
    step = .5 #step size
    gamma = 1.5 #tuning paramenter
    incoherence = 5. #incoherence of sparse matrix, i.e. smoothness
    a1 = scipy.sparse.random(d1,d2, alpha/2.)*rank*5./np.sqrt(d1*d2)
    #a1 and a2 are the corruptions, negative and positive
    a2 = scipy.sparse.random(d1,d2, alpha/2.)*rank*-5./np.sqrt(d1*d2)
    a = a1+a2
    g = rn.randn(d1,rank)/np.sqrt(d1)
    h = rn.randn(rank,d2)/np.sqrt(d2)
    Y = np.dot(g,h)+a #test on random low rank corrupted matrix
    U, V = FRPCA(Y, alpha, gamma, step, incoherence, rank, T)
    return Y, U, V, a
    
if __name__ == "__main__":
    Y, U, V, a = main()

