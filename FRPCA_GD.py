import csv
import numpy as np
import numpy.linalg as la
import timeit
import scipy
import numpy.random as rn

def Talpha(Ydat, alpha, d1, d2):
    #d1 is the number of rows
    Y = abs(Ydat)
    n1 = np.floor(alpha*d1).astype(int)
    n2 = np.floor(alpha*d2).astype(int)
    rows = np.partition(Y, -n2, axis = 1)[:, -n2]
    cols = np.partition(Y, -n1, axis = 0)[-n1]
    collim = np.dot(np.ones([d1,1]), cols)
    rowlim = np.dot(rows, np.ones([1,d2]))
    limiting = np.maximum(collim, rowlim)
    TY = np.multiply(np.multiply(Y,(Y >= limiting)),np.sign(Ydat))
    return TY

def UVproj(A, bound, d):
    for i in xrange(d):
        rownorm = la.norm(A[i])
        if rownorm > bound:
#            print(rownorm)
#            print(bound)
#            print(A[i])
            A[i] = A[i]*bound/rownorm
#            print(A[i])
#            raw_input('')
    return A
            

def FRPCA(Y, alpha, gamma, eta, mu, r, T):
    d1 = np.size(Y[:,0])
    d2 = np.size(Y[0])
    Sinit = Talpha(Y, alpha, d1, d2)
    Yrem = Y - Sinit
    [L, sigma, R] = la.svd(Yrem)
    L = L[:,0:r]
    sigmaroot = np.sqrt(sigma[0:r])
    opnorm = sigmaroot[0]
    R = R[0:r]
    U0 = np.multiply(L,sigmaroot)
    V0 = np.multiply(sigmaroot,np.transpose(R))
    Ubound = np.sqrt(2.*mu*r/d1)*opnorm
#    print(Ubound, 'ubound')
    Vbound = np.sqrt(2.*mu*r/d2)*opnorm
#    print(Vbound)
    Ut = U0
    Vt = V0
    gamalph = gamma*alpha
    Ynorm = la.norm(Y)
    for i in xrange(T):
#        print(Ut)
#        print(Vt)
        Yt = np.dot(Ut, np.transpose(Vt))
        St = Talpha(Y - Yt, gamalph, d1, d2)
#        print(St, 'St')
        lossmat = Yt+St-Y
        error = la.norm(lossmat)/Ynorm
        print(error, 'error at iteration', i)
#        print(lossmat, 'lossmat')
        UminV = np.dot(np.transpose(Ut), Ut) - np.dot(np.transpose(Vt), Vt)
#        print(UminV,'umv')
#        print(np.dot(np.transpose(Ut), Ut))
#        print(np.dot(np.transpose(Vt), Vt))
        Utnew = Ut - eta*np.dot(lossmat, Vt)-.5*eta*np.dot(Ut, UminV)
        Vtnew = Vt - eta*np.dot(np.transpose(lossmat), Ut)-.5*eta*np.dot(Vt, -UminV)
#        print(Utnew, 'unew')
#        print(Vtnew, 'vnew')
        Ut = UVproj(Utnew, Ubound, d1)
        Vt = UVproj(Vtnew, Vbound, d2)
#        raw_input('a')
    return Ut, Vt

def main():
    rank = 5
    d1 = 500
    d2 = 600
    alpha = .1
    a1 = scipy.sparse.random(d1,d2, alpha/2)*rank*5./np.sqrt(d1*d2)
    a2 = scipy.sparse.random(d1,d2, alpha/2)*rank*-5./np.sqrt(d1*d2)
    a = a1+a2
    g = rn.randn(d1,rank)/np.sqrt(d1)
    h = rn.randn(rank,d2)/np.sqrt(d2)
    Y = np.dot(g,h)+a
    U, V = FRPCA(Y, alpha, 1.5, .5, 5., rank, 30)
    return Y, U, V, a
    
if __name__ == "__main__":
    Y, U, V, a = main()

