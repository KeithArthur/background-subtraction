import csv
import numpy as np
import numpy.linalg as la
import timeit
import scipy
import numpy.random as rn

def Talpha(Y, alpha, d1, d2):
    #d1 is the number of rows
    n1 = np.floor(alpha*d1).astype(int)
    n2 = np.floor(alpha*d2).astype(int)
    rows = np.partition(Y, -n2, axis = 1)[:, -n2]
    cols = np.partition(Y, -n1, axis = 0)[-n1]
    collim = np.dot(np.ones([d1,1]), cols)
    rowlim = np.dot(rows, np.ones([1,d2]))
    limiting = np.maximum(collim, rowlim)
    TY = np.multiply(Y,(Y >= limiting))
    return TY

def UVproj(A, bound, d):
    for i in xrange(d):
        rownorm = la.norm(A[i])
        if rownorm > bound:
            A[i] = A[i]*bound/rownorm
    return A
            

def FRPCA(Y, alpha, gamma, eta, mu, T):
    d1 = np.size(Y[:,0])
    d2 = np.size(Y[0])
    Sinit = Talpha(Y, alpha, d1, d2)
    Yrem = Y - Sinit
    r = la.matrix_rank(Y)
    [L, sigma, R] = la.svd(Yrem)
    L = L[:,0:r]
    sigmaroot = np.sqrt(sigma[0:r])
    opnorm = sigmaroot[0]
    R = R[0:r]
    U0 = np.multiply(L,sigmaroot)
    V0 = np.multiply(sigmaroot,np.transpose(R))
    Ubound = np.sqrt(2*mu*r/d1)*opnorm
    Vbound = np.sqrt(2*mu*r/d2)*opnorm
    Ut = U0
    Vt = V0
    gamalph = gamma*alpha
    for i in xrange(T):
        St = Talpha(Y - np.dot(Ut, np.transpose(Vt)), gamalph, d1, d2)
        lossmat = np.dot(Ut, np.transpose(Vt))+St-Y
        UminV = np.dot(np.transpose(Ut), Ut) - np.dot(np.transpose(Vt), Vt)
        Utnew = Ut - eta*np.dot(lossmat, Vt)-.5*eta*np.dot(Ut, UminV)
        Vtnew = Vt - eta*np.dot(np.transpose(lossmat), Ut)-.5*eta*np.dot(Vt, -UminV)
        Ut = UVproj(Utnew, Ubound, d1)
        Vt = UVproj(Vtnew, Vbound, d2)
    return Ut, Vt

def main():
    a = scipy.sparse.random(100,300, .02)
    g = rn.rand(100,1)
    h = rn.rand(1,300)
    Y = np.dot(g,h)+a
    U, V = FRPCA(Y, .04, 3, .01, 10, 100)
    return Y, U, V, a
    
if __name__ == "__main__":
    Y, U, V, a = main()

