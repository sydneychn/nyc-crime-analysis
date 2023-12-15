import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix

def nearestCentroid(centroids, point):
    distances = []
    for centroid in centroids:
        dist = np.linalg.norm(centroid - point)
        distances.append(dist)
    min_index = np.argmin(distances)
    return min_index #returns index of minimum centroid in original list of centroids


def createClusters(centroids, points):
    clusters =  [[] for _ in centroids]
    for point in points:
        nearestCentroidIndex = nearestCentroid(centroids,point)
        clusters[nearestCentroidIndex].append(point)
    clusters = [np.array(cluster) for cluster in clusters] #convert all lists to ndarrays
    return clusters


def newCentroids ( clusters):
    ret_val = []
    for cluster in clusters:
        mean = np.mean(cluster, axis=0)
        ret_val.append(mean)
    return np.array(ret_val)


def Kmeans(X_train, K):
    #TO-DO PART 1
    """
    :type X_train: numpy.ndarray
    :type N: int
    :type K: int
    :rtype: List[numpy.ndarray]
    """
    numOfClusters = K

    centroids = X_train.copy()
    np.random.shuffle(centroids)
    centroids = centroids[:numOfClusters]

    while True:
        clusters = createClusters(centroids, X_train)

        new_centroids = newCentroids(clusters)

        #if converge return the clusters
        if np.array_equal(new_centroids, centroids):
            return clusters
        
        centroids = new_centroids.copy()



def testKMeansWithCSV(df):
    # df = df.sample(frac=0.1, random_state=42)
    filtered_df = df[(df['Latitude'].notna()) & (df['Longitude'].notna()) & (df['Latitude'] != 0) & (df['Longitude'] != 0)]
    last_two_columns_array = filtered_df.iloc[:, -2:].values
    print(last_two_columns_array)

    K = 5
    result_clusters = Kmeans(last_two_columns_array, K)

    # Calculate the minimum and maximum values for x and y coordinates in your data
    min_x, max_x = np.min(last_two_columns_array[:, 0]), np.max(last_two_columns_array[:, 0])
    min_y, max_y = np.min(last_two_columns_array[:, 1]), np.max(last_two_columns_array[:, 1])

    # Set the x and y axis limits to zoom in on the data
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # Plot the clusters and centroids
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for i, cluster in enumerate(result_clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f'Cluster {i+1}')

    # Plot centroids
    centroids = np.array([np.mean(cluster, axis=0) for cluster in result_clusters])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')

    plt.title('Clusters of Crime based on Geolocation')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.legend()
    plt.show()

