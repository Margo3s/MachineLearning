
import sys
import numpy
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# for loading the class MyGaussian from package machine_learning located in the directory ../ml4bda
sys.path.append( "../ml4bda" )
from machine_learning import MyGaussian

from sklearn import datasets

print( "Loading ... " )
iris = datasets.load_iris()
X = iris.data#[:, :2]  # we only take the first two features.
Y = iris.target 

print( "Transforming  ... " )

X_test = X[135:]
y_test = Y[135:]

X_train = X[:135]
y_train = Y[:135]

# Students must test with and without standard scaling
standard_scaling=True
if standard_scaling:
    norm = StandardScaler()
    norm.fit( X_train )
    X_train = norm.transform( X_train )
    X_test  = norm.transform( X_test  )

# Students must test different values of n_components
# Which is the optimal value?
# How standard scaling affects?
#pca = None
#pca = PCA( n_components = 2 )
pca = PCA( n_components = 0.7, svd_solver='full' )
if pca is not None:
    pca.fit( X_train )
    X_train = pca.transform( X_train )
    X_test  = pca.transform( X_test  )

print( "Learning  ... " )

num_cv = 10
size_cv = len(X_train)//num_cv
best_gnb = None
best_accuracy = 0.0
avg_accuracy = 0.0

for trial in range(num_cv):
    print( "\t\tCross validation trial %02d ... " % (trial+1) )

    from_sample = trial * size_cv
    to_sample = from_sample + size_cv

    # Extract the validation set
    X_validation = X_train[from_sample:to_sample]
    y_validation = y_train[from_sample:to_sample]

    # Extract the training set by means of a mask for do not use the validation set
    mask = numpy.ones( len(X_train), dtype=bool )
    mask[from_sample:to_sample] = False
    X_train_cv = X_train[ mask ]
    y_train_cv = y_train[ mask ]

    print( X_train_cv.shape )
    print( X_validation.shape )

    # Students must test the following three variants for modeling each class
    #gnb = GaussianNB()
    #gnb = MyGaussian( covar_type='diag' )
    #gnb = MyGaussian( covar_type='full' )
    gnb = MyGaussian( covar_type='tied_diag' )
    gnb.fit( iris.data, iris.target )
    y_pred = gnb.predict( X_validation )
    accuracy = ( 100.0 * (y_validation == y_pred).sum() ) / len(y_validation)
    print( "\t\tAccuracy = %.1f%%" % accuracy )

    avg_accuracy += accuracy

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_gnb = gnb

avg_accuracy /= num_cv
print( "\nAccuracy in average = %.1f%%\n" % avg_accuracy )

print( "Predicting  ... " )

y_pred = best_gnb.predict( X_test )

print( "A total of %d samples out of %d are assigned to the wrong class." % ( (y_test != y_pred).sum(), len(y_test) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (y_test == y_pred).sum() ) / len(y_test) ) )
