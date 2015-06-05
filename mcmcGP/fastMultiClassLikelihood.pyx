#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free
import scipy.linalg.blas

FLOATTYPE = np.float64
ctypedef np.float64_t FLOATTYPE_t
INTTYPE = np.int64
ctypedef np.int64_t INTTYPE_t


cdef double twoPiToMinusHalf = 1./np.sqrt(2. * np.pi )
cdef double twoToMinusHalf = 1./np.sqrt(2.)
cdef double sqrt_pi = np.sqrt(np.pi)
cdef double sqrt_2 = np.sqrt(2)

cdef extern from "math.h":
    double exp(double x) nogil

cdef extern from "math.h":
    double erf(double x) nogil

cdef double normpdf(double x) nogil:
    return twoPiToMinusHalf*exp(-x*x/2.)

cdef double normcdf(double x) nogil :
    return 0.5 + 0.5*erf(x * twoToMinusHalf )

cdef void updateXs( double*  X, double[:] gh_x, double categoryMean, double categoryStandardDeviation, int nGaussHermitePoints) nogil :
    cdef int gaussHermiteIndex
    for gaussHermiteIndex in range(nGaussHermitePoints): 
        X[gaussHermiteIndex] = gh_x[gaussHermiteIndex] * sqrt_2 * categoryStandardDeviation + categoryMean

cdef double dot(double* a, double[:] b, int N) nogil:
    cdef double ret = 0
    cdef int i
    for i in range(N):
        ret += a[i]*b[i]
    return ret

cdef void prod_in_place(double* A, double * out, int N, int M) nogil:
    cdef int n, m
    for n in range(N):
        out[n] = 1.0
        for m in range(M):
            out[n] = out[n] * A[n*M+m]

cdef void updateCdfsEpsilons( double*  X, double* cdfs, double* epsilons, double[:,:] means,  double[:,:] sDevs, int pointIndex, int nGaussHermitePoints, int P, int currentCategoryIndex) nogil:
    cdef int gaussHermiteIndex, categoryIndex   
    for gaussHermiteIndex in range( nGaussHermitePoints ):
        for categoryIndex in range( P ): 
            epsilons[gaussHermiteIndex*P + categoryIndex] = (X[gaussHermiteIndex] - means[pointIndex, categoryIndex])/sDevs[pointIndex,categoryIndex] 
            if categoryIndex==currentCategoryIndex: 
                cdfs[gaussHermiteIndex*P + categoryIndex] = 1. 
            else:
                cdfs[gaussHermiteIndex*P + categoryIndex] = normcdf( epsilons[gaussHermiteIndex*P + categoryIndex] ) 

def likelihoodQuadrature( double[:,:] means, double[:,:] variances, long[:] categories, double delta, double[:] gh_w, double[:] gh_x ):
        
    cdef double[:,:] sDevs = np.sqrt( variances )
    
    cdef int nGaussHermitePoints = gh_w.shape[0]
    
    cdef int N = means.shape[0]
    cdef int P = means.shape[1]
    
    cdef double[:] expectedLogLikelihoods = np.zeros( N )
    
    cdef double logOneMinusDelta = np.log( 1. - delta )
    cdef double logDeltaOverKminusOne = np.log( delta / (P-1.) )
    
    cdef double[:,:] gradMeans = np.zeros( (N,P) )
    cdef double[:,:] gradVariances = np.zeros( (N,P) )
    
    cdef int pointIndex, gaussHermiteIndex, categoryIndex, categoryIndexB
    cdef long currentCategoryIndex
      
    cdef double prob, intMean, intVariances, currentMean, currentStandardDeviation, currentVariance, normpdf_store
    cdef double gradDelta = 0.
    cdef double currentMeanDerivativeTerm = 0.
    cdef double currentVarianceDerivativeTerm = 0.
    
    
    with nogil, parallel():
        #allocate memory
        X = <double*> malloc (sizeof(double)*nGaussHermitePoints)
        prodCdfs = <double*> malloc (sizeof(double)*nGaussHermitePoints)
        epsilons = <double*> malloc (sizeof(double)*nGaussHermitePoints*P)   
        cdfs = <double*> malloc (sizeof(double)*nGaussHermitePoints*P)    
        meanProds = <double*> malloc (sizeof(double)*nGaussHermitePoints)
        varianceProds = <double*> malloc (sizeof(double)*nGaussHermitePoints)
        categoryMeanTerms = <double*> malloc (sizeof(double)*nGaussHermitePoints)
        categoryVarianceTerms = <double*> malloc (sizeof(double)*nGaussHermitePoints)      
        

        #loop over data
        for pointIndex in prange(N):
            currentCategoryIndex = categories[pointIndex]
            currentMean = means[pointIndex,currentCategoryIndex]
            currentStandardDeviation = sDevs[pointIndex,currentCategoryIndex]
            currentVariance = variances[pointIndex,currentCategoryIndex]
    
            updateXs( X, gh_x, currentMean, currentStandardDeviation, nGaussHermitePoints )
            updateCdfsEpsilons( X, cdfs, epsilons, means, sDevs, pointIndex, nGaussHermitePoints, P, currentCategoryIndex )
            prod_in_place(cdfs, prodCdfs, nGaussHermitePoints, P)

            prob = dot(prodCdfs, gh_w, nGaussHermitePoints) / sqrt_pi
            expectedLogLikelihoods[pointIndex] = logOneMinusDelta * prob + logDeltaOverKminusOne * (1.-prob) 
            gradDelta += (1.-prob)/ delta - prob/(1.-delta) 
                        
            #Now compute derivatives.
            for categoryIndex in range(P): 
                if categoryIndex == currentCategoryIndex: #this has special behaviour 
                    for gaussHermiteIndex in range( nGaussHermitePoints ):
                        currentMeanDerivativeTerm = (X[gaussHermiteIndex] - currentMean)/currentVariance 
                        currentVarianceDerivativeTerm = 0.5*(X[gaussHermiteIndex] - currentMean)**2 / currentVariance**2 - 0.5/currentVariance
                        meanProds[gaussHermiteIndex] = currentMeanDerivativeTerm * prodCdfs[gaussHermiteIndex]
                        varianceProds[gaussHermiteIndex] = currentVarianceDerivativeTerm * prodCdfs[gaussHermiteIndex]
                else:
                    for gaussHermiteIndex in range( nGaussHermitePoints ): 
                        meanProds[gaussHermiteIndex] = 1.
                        varianceProds[gaussHermiteIndex] = 1.
                        normpdf_store  = normpdf(epsilons[gaussHermiteIndex*P + categoryIndex])
                        categoryMeanTerms[gaussHermiteIndex] = -1.*normpdf_store / sDevs[pointIndex,categoryIndex] 
                        categoryVarianceTerms[gaussHermiteIndex] = -0.5*normpdf_store*epsilons[gaussHermiteIndex *P + categoryIndex] / sDevs[pointIndex,categoryIndex]**2
                                        
                    for categoryIndexB in range(P):
                        if categoryIndexB==categoryIndex:
                            for gaussHermiteIndex in range( nGaussHermitePoints ):
                                meanProds[gaussHermiteIndex] = meanProds[gaussHermiteIndex] * categoryMeanTerms[gaussHermiteIndex]
                                varianceProds[gaussHermiteIndex] = varianceProds[gaussHermiteIndex] * categoryVarianceTerms[gaussHermiteIndex]
                        else: 
                            for gaussHermiteIndex in range( nGaussHermitePoints ):
                                meanProds[gaussHermiteIndex] = meanProds[gaussHermiteIndex] * cdfs[gaussHermiteIndex*P+categoryIndexB]
                                varianceProds[gaussHermiteIndex] = varianceProds[gaussHermiteIndex] * cdfs[gaussHermiteIndex*P+categoryIndexB]
                    
                intMean = dot(meanProds , gh_w, nGaussHermitePoints) / sqrt_pi
                intVariances = dot(varianceProds , gh_w , nGaussHermitePoints) / sqrt_pi
                gradMeans[pointIndex,categoryIndex] = intMean * logOneMinusDelta - intMean * logDeltaOverKminusOne
                gradVariances[pointIndex,categoryIndex] = intVariances * logOneMinusDelta - intVariances * logDeltaOverKminusOne 
        #end of prange

        free(X)
        free(prodCdfs)
        free(epsilons)
        free(cdfs)
        free(meanProds)
        free(varianceProds)
        free(categoryMeanTerms)
        free(categoryVarianceTerms)
            
    return np.asarray(expectedLogLikelihoods), np.asarray(gradMeans), np.asarray(gradVariances), np.asarray(gradDelta)

def predictiveQuadrature( double[:,:] means, double[:,:] variances, double delta, double[:] gh_w, double[:] gh_x ): 
    cdef double[:,:] sDevs = np.sqrt( variances )

    cdef Py_ssize_t nGaussHermitePoints = gh_w.shape[0]
    cdef Py_ssize_t N = means.shape[0]
    cdef Py_ssize_t P = means.shape[1]

    cdef double oneMinusDelta = 1. - delta
    cdef double deltaOverKminusOne = delta / (P-1.)

    cdef double[:,:]  predictiveProbabilities = np.zeros( (N,P ) )
    
    cdef Py_ssize_t pointIndex, categoryIndex
    cdef double prob = 0.
    cdef double currentMean = 0.
    cdef double currentStandardDeviation = 0.
    cdef double currentVariance = 0.

    with nogil, parallel():
        X = <double*> malloc (sizeof(double)*nGaussHermitePoints)
        prodCdfs = <double*> malloc (sizeof(double)*nGaussHermitePoints)
        epsilons = <double*> malloc (sizeof(double)*nGaussHermitePoints*P) 
        cdfs = <double*> malloc (sizeof(double)*nGaussHermitePoints*P) 
        for pointIndex in prange(N):     
            for categoryIndex in range(P):            
                currentMean = means[pointIndex,categoryIndex]
                currentStandardDeviation = sDevs[pointIndex,categoryIndex]
                currentVariance = variances[pointIndex,categoryIndex]            
                updateXs( X, gh_x, currentMean, currentStandardDeviation, nGaussHermitePoints )
                updateCdfsEpsilons( X, cdfs, epsilons, means, sDevs, pointIndex, nGaussHermitePoints, P, categoryIndex )
                prod_in_place(cdfs, prodCdfs, nGaussHermitePoints, P)
                prob = dot( prodCdfs, gh_w, nGaussHermitePoints ) / sqrt_pi
                predictiveProbabilities[pointIndex, categoryIndex] = oneMinusDelta *prob + deltaOverKminusOne * (1. - prob )
        free(X)
        free(prodCdfs)            
        free(epsilons)
        free(cdfs)
            
    return predictiveProbabilities
            
