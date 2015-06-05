import numpy as np
import cPickle

def getSamples( nSamples, rng ):
    #coordinates are (x,y)
    meanR = np.array( [0. , 0.5] )
    meanG = np.array( [-1.5, 0.] )
    meanB = np.array( [1.5, 0.] )
    
    covR = np.array( [[ 5., 0], [0., 0.1] ] )
    covG = np.array( [[ 0.05, 0], [0, 3. ] ] )
    transformMatrix = np.array( [[ 1., 1.], [-1., 1.] ] ) / np.sqrt(2.)
    diagonalMatrix = np.diag( [3., 0.1 ] )
    covB = np.dot( transformMatrix.T , np.dot( diagonalMatrix  , transformMatrix ) )
    
    LR = np.linalg.cholesky( covR )
    LG = np.linalg.cholesky( covG )
    LB = np.linalg.cholesky( covB )
    
    means = [ meanR, meanG, meanB ]
    Ls = [ LR, LG, LB ]

    samples = np.zeros( (nSamples,2) )
    categories = np.zeros( nSamples , dtype='int' )

    for pointIndex in range( nSamples ):
        currentPointCategory = rng.choice( 3 )
        samples[pointIndex,:] = means[currentPointCategory] + np.dot(Ls[currentPointCategory],rng.randn(2) )
        categories[pointIndex] = currentPointCategory
        
    return samples, categories

if __name__ == '__main__':
    nTrainingSamples = 750
    nTestingSamples = 750
    
    rng = np.random.RandomState( 1 )
    X_train, Y_train = getSamples( nTrainingSamples, rng )
    X_test, Y_test = getSamples( nTrainingSamples, rng )
    
    from matplotlib import pylab as plt
    RTrain = np.array( [ X_train[index,:] for index in range(X_train.shape[0] ) if Y_train[index]== 0 ] )
    GTrain = np.array( [ X_train[index,:] for index in range(X_train.shape[0] ) if Y_train[index]== 1 ] )
    BTrain = np.array( [ X_train[index,:] for index in range(X_train.shape[0] ) if Y_train[index]== 2 ] )
    
    plt.plot( RTrain[:,0], RTrain[:,1] ,'ro')
    plt.plot( GTrain[:,0], GTrain[:,1] , 'bo')
    plt.plot( BTrain[:,0], BTrain[:,1] , 'go')
    plt.xlim( [-6.,6.] )
    plt.ylim( [-6.,6.] )
    
    data = {'X_train':X_train,'Y_train':Y_train,'X_test':X_test,'Y_test':Y_test}
    cPickle.dump( data , open('data','w') )

    from matplotlib2tikz import save as save_tikz
    save_tikz('simple_data.tikz')
    
