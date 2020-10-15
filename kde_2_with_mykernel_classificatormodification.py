
import sys
import numpy
from matplotlib import pyplot

sys.path.append( "../ml4bda" )
from machine_learning import MyKernel
from machine_learning import MyKernelClassifier
from machine_learning import generate_datasets



X_train,Y_train,X_test,Y_test = generate_datasets.generate_multivariate_normals( 5, 5, 150, 150, 5.0, 2.0 )

targets=numpy.unique(Y_train)
num_classes = len(targets)
samples_per_class = []
for c in range(num_classes):
    samples_per_class.append( X_train[ Y_train==targets[c] ] )
    print( "Class %3d with %10d " % ( (c+1), len(samples_per_class[c]) ) )

kernel='exponential'
estimators=[]

for k in range(num_classes):
    estimators.append( MyKernel().fit(samples_per_class[k]) )

y_pred = numpy.zeros( len(X_test), dtype=type(targets[c]) )
best_log_dens = numpy.zeros( len(X_test) )
for k in range(num_classes):
    print( "Scoring test samples for class %3d " % k )
    log_dens = estimators[k].score_samples(X_test)
    if 0 == k :
        best_log_dens[:] = log_dens[:]
        y_pred[:] = 0
    else:
        print( "Classifying test samples up to class %3d " % k )
        for n in range(len(X_test)):
            print( "\r %10d " % n, end='' )
            if log_dens[n] > best_log_dens[n]:
                best_log_dens[n] = log_dens[n]
                y_pred[n] = targets[k]
        print( " " )
    
print( "A total of %d samples out of %d are assigned to the wrong class." % ( (Y_test != y_pred).sum(), len(Y_test) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test) ) )


for h in range(2,20):
    for k in range(num_classes):
        estimators.append( MyKernel(h).fit(samples_per_class[k]) )

    y_pred = numpy.zeros( len(X_test), dtype=type(targets[c]) )
    best_log_dens = numpy.zeros( len(X_test) )
    for k in range(num_classes):
        print( "Scoring test samples for class %3d " % k )
        log_dens = estimators[k].score_samples(X_test)
        if 0 == k :
            best_log_dens[:] = log_dens[:]
            y_pred[:] = 0
        else:
            print( "Classifying test samples up to class %3d " % k )
            for n in range(len(X_test)):
                print( "\r %10d " % n, end='' )
                if log_dens[n] > best_log_dens[n]:
                    best_log_dens[n] = log_dens[n]
                    y_pred[n] = targets[k]
            print( " " )

    print( "h = %d" % h )
    print( "A total of %d samples out of %d have been assigned to a wrong class." % ( (Y_test != y_pred).sum(), len(Y_test) ), end='  ' )
    print( "Accuracy = %.1f%%" % ( ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test) ) )

for k in range(num_classes):
    estimators.append( MyKernel(h).fit(samples_per_class[k]) )

y_pred = numpy.zeros( len(X_test), dtype=type(targets[c]) )
best_log_dens = numpy.zeros( len(X_test) )
for k in range(num_classes):
    print( "Scoring test samples for class %3d " % k )
    log_dens = estimators[k].score_samples(X_test)
    if 0 == k :
        best_log_dens[:] = log_dens[:]
        y_pred[:] = 0
    else:
        print( "Classifying test samples up to class %3d " % k )
        for n in range(len(X_test)):
            print( "\r %10d " % n, end='' )
            if log_dens[n] > best_log_dens[n]:
                best_log_dens[n] = log_dens[n]
                y_pred[n] = targets[k]
        print( " " )
print( "A total of %d samples out of %d are assigned to the wrong class." % ( (Y_test != y_pred).sum(), len(Y_test) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test) ) )

