
# Practica 4

import numpy
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from load_mnist import load_mnist # if this line fails, do this: ln -s ../practica02/load_mnist.py .

print( "Loading ... " )

X,y = load_mnist()

print( "Transforming  ... " )

X_train = X[:60000]
y_train = y[:60000]
X_test  = X[60000:]
y_test  = y[60000:]

# These three hyper-parameters should be changed by students to observe their effects
standard_scaling=False
min_max_scaling=False
pca_components=55
#
if pca_components > 0:
    pca = PCA( n_components=pca_components )
    pca.fit( X_train )
    X_train = pca.transform( X_train )
    X_test  = pca.transform( X_test  )

if standard_scaling:
    norm = StandardScaler()
    norm.fit( X_train )
    X_train = norm.transform( X_train )
    X_test  = norm.transform( X_test  )
elif min_max_scaling:
    norm = MinMaxScaler()
    norm.fit( X_train )
    X_train = norm.transform( X_train )
    X_test  = norm.transform( X_test  )
    

print( "Selecting  ... " )

targets=numpy.unique(y)
num_classes = len(targets)
samples_per_class = []
for c in range(num_classes):
    samples_per_class.append( X_train[ y_train==targets[c] ] )
    print( samples_per_class[-1].shape )

print( "Learning  phase 1 :: GMM per class ... " )

# Covariance valid types: 'diag', 'spherical', 'tied', 'full'

# This hyper-parameter should be changed by stduents to observe its effect -- in combination with the other hyper-parameters
num_subclases=10
mixtures = []
for c in range(num_classes):
    print( "                     GMM for class %2d ... " % c )
    mixtures.append( GaussianMixture( n_components=num_subclases, covariance_type='full',
                            init_params='random', max_iter=200, n_init=10 ) )
    mixtures[c].fit( samples_per_class[c] )

print( "Working with %d components per GMM" % num_subclases )

print( "Predicting  ... " )

densities = numpy.zeros( [ len(X_test), num_classes ] )
prioris = numpy.zeros(num_classes)

"""
    According to the Bayes' rule:
        P(c) = A priori probability of classs c
        p(x|c) = Conditional probability density of observing the sample x when the state of the system corresponds to class c
        P(c|x) = P(c)*p(x|c) / p(x) -- A posteriori probability that the system is in state c when the sample x has been observed
        p(x) = Likelihood of sample x computed as the sumatory of all P(k)*p(x|k) for all the classes, not used here for classifying
"""

for c in range(num_classes):
    # P(c) 
    prioris[c] = numpy.log( len(samples_per_class[c]) ) - numpy.log( len(X_train) ) # work with logs for robust computations
    # p(x|c)
    densities[:,c] = mixtures[c].score_samples( X_test )

y_pred = numpy.zeros(len(X_test),dtype=type(targets[0]))
for n in range(len(X_test)):
    k = 0
    for c in range(num_classes):
        #       p(x|c)    *    P(c)    >      p(x|k)    *    P(k)
        if densities[n,c] + prioris[c] > densities[n,k] + prioris[k] : k = c
    y_pred[n] = targets[k]


print( "A total of %d samples out of %d are assigned to a wrong class." % ( (y_test != y_pred).sum(), len(y_test) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (y_test == y_pred).sum() ) / len(y_test) ) )
