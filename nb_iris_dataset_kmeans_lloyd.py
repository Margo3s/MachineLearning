
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import time
import math
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
from sklearn import metrics

sys.path.append( "../ml4bda" )
from machine_learning import KMeans


class AnimatedClustering(object):

    def __init__( self, X ):
        #
        # Study the behaviour of the algorithm for different number of clusters
        #
        self.num_clusters=3
        self.X_ = X
        self.data_ = numpy.zeros( (2, X.shape[0]+self.num_clusters) )
        self.data_[0,:len(X)] = X[:,0]
        self.data_[1,:len(X)] = X[:,1]
        #
        # Check the behaviour of the Lloyd algorithm with random initialization and with Katsavounidis
        #
        #self.kmeans = KMeans( self.num_clusters, init='random', max_iter=1 )
        self.kmeans = KMeans( self.num_clusters, init='Katsavounidis', max_iter=1 )
        self.Y_ = numpy.ones( len(X), dtype='int' )

        self.fig_, self.ax_ = pyplot.subplots()
        self.ani_ = animation.FuncAnimation( self.fig_, self.update_figure, self.generate_data, init_func=self.setup_plot, interval=1000, blit=True, repeat=False )
        self.changes_=0

    def setup_plot( self ):
        self.colour_ = numpy.ones( len(self.X_)+self.num_clusters )*3
        self.sizes_ = numpy.ones( self.X_.shape[0]+self.num_clusters ) * 30
        self.colour_[len(self.X_):] = self.num_clusters+1
        self.sizes_[len(self.X_):] = 100
        self.scat_ = self.ax_.scatter( self.data_[0,:], self.data_[1,:], c=self.colour_, s=self.sizes_, marker='o', edgecolors='none', animated=False )
        return self.scat_,

    def generate_data( self ):
        self.changes_ = len(self.X_)
        self.kmeans.fit( self.X_ )
        #self.kmeans.lloyd( self.X_, num_iter=1 )
        self.colour_[:len(self.X_)] = self.kmeans.predict( self.X_ )
        self.data_[0,len(self.X_):] = self.kmeans.cluster_centers_[:,0]
        self.data_[1,len(self.X_):] = self.kmeans.cluster_centers_[:,1]
        yield self.data_, self.colour_, self.sizes_

        while self.changes_ > 0 :
            self.changes_ = self.kmeans.fit_iteration( self.X_ )
            self.Y_[:] = self.kmeans.predict( self.X_ )
            self.colour_[:len(self.Y_)] = self.Y_[:]
            self.data_[0,len(self.X_):] = self.kmeans.cluster_centers_[:,0]
            self.data_[1,len(self.X_):] = self.kmeans.cluster_centers_[:,1]
            yield self.data_, self.colour_, self.sizes_
            
        #Averiguar clase de cada centroide
        iterator=0
        positions_center0=[]
        positions_center1=[]
        positions_center2=[]
        for center in self.colour_[:len(self.X_)]:          
            if center == 0:
                positions_center0.append(Y[iterator])
            elif center == 1:
                positions_center1.append(Y[iterator])
            else:
                positions_center2.append(Y[iterator])
            iterator=iterator+1
        print('clases reales del centroide 0')
        print(positions_center0)
        print('clases reales del centroide 1')
        print(positions_center1)
        print('clases reales del centroide 2')
        print(positions_center2)
        
        counts0 = numpy.bincount(positions_center0)
        print('Clase más repetida para el centroide 0:')
        clase_centroide0=numpy.argmax(counts0)
        print(clase_centroide0)
        error=0
        for clase_de_muestra in positions_center0:
            if(clase_de_muestra != clase_centroide0):
                error=error+1
                
        counts1 = numpy.bincount(positions_center1)
        print('Clase más repetida para el centroide 1:')
        clase_centroide1=numpy.argmax(counts1)
        print(clase_centroide1)
        
        for clase_de_muestra in positions_center1:
            if(clase_de_muestra != clase_centroide1):
                error=error+1        
        
        counts2 = numpy.bincount(positions_center2)
        print('Clase más repetida para el centroide 2:')
        clase_centroide2=numpy.argmax(counts2)
        print(clase_centroide2)        

        for clase_de_muestra in positions_center2:
            if(clase_de_muestra != clase_centroide2):
                error=error+1 
                
        print( "A total of %d samples out of %d are assigned to the wrong class." % ( error, len(Y) ))
        print( "Accuracy = %.1f%%" % ( ( 100.0 * (len(Y)-error)) / len(Y) ) )                

    def update_figure( self, generated_data ):
        data,colour,sizes = generated_data
        print( "clusters = %d changes = %12d   J = %20.8f  %.8f" % ( self.num_clusters, self.changes_, self.kmeans.J, self.kmeans.improvement() ) )

        pyplot.clf()
        #pyplot.set_axis_bgcolor( 'white' )
        self.scat_ = self.ax_.scatter( self.data_[0,:], self.data_[1,:], c=colour, s=sizes, marker='o', edgecolors='none', animated=False )
        #pyplot.draw()
        return self.scat_,

    def show( self ):
        pyplot.show()


## MAIN

if __name__ == '__main__' :

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data#[:, :2]  # we only take the first two features.
    Y = iris.target
    print('Y original')
    print(Y)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    #plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=60 )
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    ac = AnimatedClustering(X)
    ac.show()
    plt.show()
    print('Clusters obtenidos')
    print( ac.kmeans.cluster_centers_ )


   



